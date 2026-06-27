"""
Google Images scraper — HTTP/2 via httpx + tenacity retry + DuckDuckGo fallback.

Phase 1: paginate Google Image search HTML to collect candidate URLs.
         Falls back to DuckDuckGo if Google bot-detects (JS redirect).
Phase 2: download images concurrently (semaphore-bounded) into a temp dir,
         skipping hashes already in the dedup store.
Results are put on a queue so the caller can filter one image at a time
while downloads continue in the background.
"""

import asyncio
import hashlib
import queue
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Generator
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from ddgs import DDGS as _DDGS
    _DDG_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS
        _DDG_OK = True
    except ImportError:
        _DDG_OK = False

_SENTINEL = object()
_SKIP_HOSTS = {"encrypted-tbn", "gstatic.com", "google.com", "googleusercontent.com"}
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

_SIZE_MAP = {"large": "isz:l", "medium": "isz:m", "icon": "isz:i"}
_TYPE_MAP = {
    "photo": "itp:photo", "clipart": "itp:clipart",
    "lineart": "itp:lineart", "face": "itp:face", "animated": "itp:animated",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
}


class _RateLimited(Exception):
    pass


class GoogleScraper:
    BASE_URL = "https://www.google.com/search"

    def __init__(
        self,
        max_per_keyword: int = 200,
        concurrency: int = 4,
        request_delay: float = 1.5,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        timeout: float = 30.0,
        image_size: str = "large",
        image_type: str = "photo",
        hash_store=None,
    ):
        self.max_per_keyword = max_per_keyword or 200
        self.concurrency = concurrency
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.image_size = image_size
        self.image_type = image_type
        self.hash_store = hash_store  # optional HashStore for pre-download dedup

    def scrape(self, keyword: str) -> Generator[str, None, None]:
        tmpdir = tempfile.mkdtemp(prefix="scrape_google_")
        result_q: queue.Queue = queue.Queue()

        def _run():
            try:
                asyncio.run(self._crawl(keyword, tmpdir, result_q))
            except Exception as exc:
                logger.error(f"[google] async error: {exc}")
            finally:
                result_q.put(_SENTINEL)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            while True:
                item = result_q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            t.join(timeout=10)
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── async core ──────────────────────────────────────────────────────────────

    async def _crawl(self, keyword: str, tmpdir: str, result_q: queue.Queue):
        async with httpx.AsyncClient(
            http2=True,
            headers=_HEADERS,
            timeout=httpx.Timeout(self.timeout, connect=self.timeout),
            limits=httpx.Limits(
                max_connections=self.concurrency,
                max_keepalive_connections=self.concurrency,
            ),
            follow_redirects=True,
        ) as client:
            candidates = await self._collect_urls(client, keyword)
            logger.info(f"[google] '{keyword}' — {len(candidates)} candidate URLs")

            sem = asyncio.Semaphore(self.concurrency)
            lock = asyncio.Lock()
            state = {"count": 0}
            seen_hashes: set[str] = set()

            tasks = [
                asyncio.create_task(
                    self._download_one(
                        client, sem, lock, url, tmpdir, state, seen_hashes, result_q
                    )
                )
                for url in candidates
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_urls(self, client: httpx.AsyncClient, keyword: str) -> list[str]:
        want = self.max_per_keyword * 3
        collected: list[str] = []
        seen: set[str] = set()
        start = 0
        blocked = False

        tbs_parts = []
        if self.image_size in _SIZE_MAP:
            tbs_parts.append(_SIZE_MAP[self.image_size])
        tbs_parts.append(_TYPE_MAP.get(self.image_type, _TYPE_MAP["photo"]))
        tbs = ",".join(tbs_parts)

        while len(collected) < want:
            params = {"q": keyword, "tbm": "isch", "hl": "en", "start": start, "tbs": tbs}
            url = f"{self.BASE_URL}?{urlencode(params)}"

            @self._retry()
            async def _fetch(u=url):
                resp = await client.get(u)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RateLimited(resp.status_code)
                resp.raise_for_status()
                return resp.text

            try:
                html = await _fetch()
            except Exception as exc:
                logger.warning(f"[google] search error (start={start}): {exc}")
                blocked = True
                break

            if "enablejs" in html[:1000]:
                logger.warning(f"[google] '{keyword}' bot-detected — trying DuckDuckGo fallback")
                blocked = True
                break

            new = _extract_urls(html)
            if not new:
                break
            for u in new:
                if u not in seen:
                    seen.add(u)
                    collected.append(u)
            start += 20
            await asyncio.sleep(self.request_delay)

        if (blocked or not collected) and _DDG_OK:
            logger.info(f"[google→ddg] falling back for '{keyword}'")
            loop = asyncio.get_event_loop()
            ddg = await loop.run_in_executor(None, self._ddg_urls, keyword, want)
            logger.info(f"[google→ddg] {len(ddg)} DDG URLs")
            for u in ddg:
                if u not in seen:
                    seen.add(u)
                    collected.append(u)
        elif blocked and not _DDG_OK:
            logger.warning("[google] DuckDuckGo not available — pip install ddgs")

        return collected

    def _ddg_urls(self, keyword: str, want: int) -> list[str]:
        _size_ddg = {"large": "Large", "medium": "Medium", "icon": "Small"}
        try:
            results = _DDGS().images(
                keyword,
                max_results=want,
                type_image=self.image_type,
                size=_size_ddg.get(self.image_size, "Large"),
            )
            return [r["image"] for r in results if r.get("image")]
        except Exception as exc:
            logger.warning(f"[ddg] search failed: {exc}")
            return []

    async def _download_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        lock: asyncio.Lock,
        url: str,
        tmpdir: str,
        state: dict,
        seen_hashes: set,
        result_q: queue.Queue,
    ):
        async with sem:
            async with lock:
                if state["count"] >= self.max_per_keyword:
                    return

            @self._retry()
            async def _fetch():
                resp = await client.get(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RateLimited(resp.status_code)
                if resp.status_code != 200:
                    return None
                ct = resp.headers.get("content-type", "")
                if "image" not in ct:
                    return None
                return resp.content, ct

            try:
                result = await _fetch()
            except Exception as exc:
                logger.debug(f"[google] download failed {url[:60]}: {exc}")
                return

            if result is None:
                return
            data, ct = result

            h = hashlib.blake2b(data, digest_size=20).hexdigest()

            async with lock:
                if h in seen_hashes:
                    return
                if self.hash_store and not self.hash_store.is_new(data):
                    logger.debug(f"[google] skip (known hash): {url[:60]}")
                    return
                if state["count"] >= self.max_per_keyword:
                    return
                seen_hashes.add(h)
                state["count"] += 1

            ext = _guess_ext(url, ct)
            dest = Path(tmpdir) / f"google_{h}{ext}"
            dest.write_bytes(data)
            logger.debug(f"[google] saved {dest.name} ({state['count']}/{self.max_per_keyword})")
            result_q.put(str(dest))

    def _retry(self):
        return retry(
            retry=retry_if_exception_type(_RateLimited),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30, exp_base=self.backoff_base),
            reraise=True,
        )


def _extract_urls(html: str) -> list[str]:
    regex_urls = re.findall(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp))"', html, re.I)
    soup = BeautifulSoup(html, "html.parser")
    tag_urls = [
        img.get("src") or img.get("data-src") or ""
        for img in soup.find_all("img")
    ]
    seen: set[str] = set()
    result = []
    for u in regex_urls + tag_urls:
        if not u or not u.startswith("http"):
            continue
        if any(h in u for h in _SKIP_HOSTS):
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _guess_ext(url: str, ct: str) -> str:
    for ext in _IMAGE_EXTS:
        if url.lower().endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"
