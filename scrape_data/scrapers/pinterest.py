"""
Pinterest scraper via gallery-dl.

gallery-dl handles login/cookies, pagination, and rate-limiting.
Images are downloaded to a temp dir and yielded one-by-one so the
pipeline can filter them before they accumulate on disk.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator

from loguru import logger

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class PinterestScraper:
    def __init__(
        self,
        cookies_file: str,
        max_per_keyword: int = 0,
        sleep_between_requests: float = 0.5,
        timeout: int = 3600,
        archive_path: str = "./logs/gdl_archive_pinterest.db",
    ):
        self.cookies_file = cookies_file
        self.max_per_keyword = max_per_keyword
        self.sleep_requests = sleep_between_requests
        self.timeout = timeout
        self.archive_path = archive_path
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)

    def scrape(self, keyword: str) -> Generator[str, None, None]:
        if not Path(self.cookies_file).exists():
            logger.error(f"cookies file not found: {self.cookies_file}")
            return

        query = keyword.replace(" ", "%20")
        url = f"https://www.pinterest.com/search/pins/?q={query}"

        cmd = [
            "gallery-dl",
            "--dest", "",        # overridden by -o directory=[]
            "--sleep", str(self.sleep_requests),
            "--retries", "4",
            "--download-archive", self.archive_path,
            "--filter", "extension in ('jpg', 'jpeg', 'png', 'webp')",
            "--user-agent", _USER_AGENT,
            "--cookies", self.cookies_file,
            "-o", "filename=pinterest_{id}{media_id:?_//}.{extension}",
            "-o", "directory=[]",
            "-o", "pinterest.image-size=originals",
        ]
        if self.max_per_keyword and self.max_per_keyword > 0:
            cmd += ["--range", f"1-{self.max_per_keyword}"]
        cmd.append(url)

        tmpdir = tempfile.mkdtemp(prefix="scrape_pinterest_")
        # gallery-dl respects --dest only when directory is not overridden;
        # using -o directory=[] makes it drop files directly into tmpdir.
        # Patch the --dest value after building the command.
        cmd[cmd.index("--dest") + 1] = tmpdir

        logger.debug(f"[pinterest] gallery-dl → {tmpdir}")
        yielded: set[str] = set()
        stderr_lines: list[str] = []

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        def _drain(stream, tag):
            for line in stream:
                line = line.rstrip()
                if line:
                    stderr_lines.append(line)
                    logger.debug(f"[gallery-dl/{tag}] {line}")

        threading.Thread(target=_drain, args=(proc.stdout, "out"), daemon=True).start()
        threading.Thread(target=_drain, args=(proc.stderr, "err"), daemon=True).start()

        deadline = time.time() + self.timeout
        try:
            while True:
                for f in sorted(_list_images(tmpdir) - yielded):
                    _wait_stable(f)
                    yielded.add(f)
                    yield f

                if proc.poll() is not None:
                    for f in sorted(_list_images(tmpdir) - yielded):
                        _wait_stable(f)
                        yield f
                    rc = proc.returncode
                    if rc not in (0, 1):
                        logger.warning(
                            f"[pinterest] gallery-dl exit {rc} — "
                            f"{stderr_lines[-3:] if stderr_lines else '(none)'}"
                        )
                    break

                if time.time() > deadline:
                    logger.warning(f"[pinterest] timeout after {self.timeout}s")
                    proc.terminate()
                    break

                time.sleep(0.5)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            shutil.rmtree(tmpdir, ignore_errors=True)


def _list_images(directory: str) -> set[str]:
    p = Path(directory)
    if not p.exists():
        return set()
    result = set()
    for ext in _IMAGE_EXTS:
        result.update(str(f) for f in p.rglob(f"*{ext}"))
    return result


def _wait_stable(path: str, retries: int = 6, interval: float = 0.3):
    prev = -1
    for _ in range(retries):
        try:
            size = os.path.getsize(path)
            if size == prev and size > 0:
                return
            prev = size
        except OSError:
            pass
        time.sleep(interval)
