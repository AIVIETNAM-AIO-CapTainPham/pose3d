"""
Minimal pose-image scraping pipeline.

Flow per image:
  Scrape (Pinterest / Google)
    → Blake2b dedup          (skip if already seen)
    → YOLOv11 person check   (largest bbox must be >= 1/3 of image area)
    → Save to output_dir/{keyword}/

Usage:
    uv run scrape                                    # full run via config.yaml
    uv run scrape --platform pinterest
    uv run scrape --platform google
    uv run scrape --keyword "pose on cafe"
    uv run scrape --config my_config.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import yaml
from loguru import logger

from .dedup import HashStore
from .person_filter import check_person
from .scrapers import GoogleScraper, PinterestScraper


# ── config helpers ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load keywords from separate keywords.yaml if present alongside config
    config_dir = Path(path).parent
    kw_file = config_dir / "keywords.yaml"
    if kw_file.exists():
        with open(kw_file, "r", encoding="utf-8") as f:
            kw_data = yaml.safe_load(f)
        if isinstance(kw_data, dict) and "keywords" in kw_data:
            cfg["keywords"] = kw_data["keywords"]

    return cfg


def _delete(path: str):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _setup_logging(log_dir: str, level: str = "INFO"):
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=level, colorize=True)
    log_file = Path(log_dir) / "scrape.log"
    logger.add(str(log_file), level=level, rotation="50 MB", retention=5)


# ── pipeline ─────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        paths = config["paths"]

        self.output_dir = paths["output_dir"]
        self.hash_store = HashStore(paths["hash_db"])

        p = config.get("person_filter", {})
        self.model_path    = p.get("model", "yolo11x.pt")
        self.confidence    = p.get("confidence", 0.4)
        self.min_area_ratio = p.get("min_area_ratio", 1 / 3)

        self.sleep_between = config.get("sleep_between_images", 0.5)

        self.stats = {"scraped": 0, "duplicate": 0, "no_person": 0, "saved": 0}

    def process(self, image_path: str, keyword: str) -> bool:
        self.stats["scraped"] += 1

        try:
            raw = Path(image_path).read_bytes()
        except Exception as exc:
            logger.warning(f"Cannot read {image_path}: {exc}")
            return False

        if not self.hash_store.is_new(raw):
            self.stats["duplicate"] += 1
            logger.debug(f"DUPE  {Path(image_path).name}")
            _delete(image_path)
            return False

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot decode {image_path}")
            _delete(image_path)
            return False

        passed, details = check_person(
            image,
            model_path=self.model_path,
            confidence=self.confidence,
            min_area_ratio=self.min_area_ratio,
        )

        if not passed:
            self.stats["no_person"] += 1
            reason = details.get("reason", "rejected")
            ratio  = details.get("largest_ratio", 0)
            logger.debug(f"SKIP  [{reason}] ratio={ratio:.3f}  {Path(image_path).name}")
            _delete(image_path)
            return False

        self._save(image_path, keyword, raw)
        ratio = details.get("largest_ratio", 0)
        logger.success(
            f"SAVE  [{keyword}] ratio={ratio:.3f}  {Path(image_path).name}"
        )
        self.stats["saved"] += 1
        return True

    def _save(self, image_path: str, keyword: str, raw: bytes):
        self.hash_store.add(raw)
        safe = keyword.replace(" ", "_").replace("/", "-")[:70]
        dest_dir = Path(self.output_dir) / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(image_path).name
        # Avoid name collision
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = dest_dir / f"{stem}_{self.stats['saved']}{suffix}"
        shutil.move(image_path, dest)

    def print_stats(self):
        s = self.stats
        total = s["scraped"]
        rate  = s["saved"] / total * 100 if total else 0
        print()
        print("=" * 50)
        print("  RESULTS")
        print("=" * 50)
        print(f"  Scraped     : {total:,}")
        print(f"  Duplicate   : {s['duplicate']:,}")
        print(f"  No person   : {s['no_person']:,}")
        print(f"  Saved       : {s['saved']:,}  ({rate:.1f}%)")
        print(f"  Hash DB     : {self.hash_store.count():,} entries")
        print(f"  Output dir  : {os.path.abspath(self.output_dir)}")
        print("=" * 50)
        print()

    def close(self):
        self.hash_store.close()


# ── scrapers factory ─────────────────────────────────────────────────────────────

def _make_pinterest(config: dict) -> PinterestScraper:
    pc = config.get("platforms", {}).get("pinterest", {})
    return PinterestScraper(
        cookies_file=config["paths"]["cookies"],
        max_per_keyword=pc.get("max_per_keyword", 0),
        sleep_between_requests=pc.get("sleep_between_requests", 0.5),
        timeout=pc.get("timeout", 3600),
        archive_path=str(
            Path(config["paths"].get("log_dir", "./logs"))
            / "gdl_archive_pinterest.db"
        ),
    )


def _make_google(config: dict, hash_store: HashStore) -> GoogleScraper:
    gc = config.get("platforms", {}).get("google", {})
    img_f = gc.get("image_filter", {})
    return GoogleScraper(
        max_per_keyword=gc.get("max_per_keyword", 200),
        concurrency=gc.get("max_concurrency", 4),
        request_delay=gc.get("request_delay", 1.5),
        max_retries=gc.get("max_retries", 4),
        backoff_base=gc.get("backoff_base", 2.0),
        timeout=gc.get("timeout", 30.0),
        image_size=img_f.get("size", "large"),
        image_type=img_f.get("image_type", "photo"),
        hash_store=hash_store,
    )


# ── runner ───────────────────────────────────────────────────────────────────────

def run(config: dict, platform_filter: str = "", keyword_filter: str = ""):
    pipeline = Pipeline(config)

    keywords: list[str] = config.get("keywords", [])
    platforms: dict     = config.get("platforms", {})

    for platform_name, plat_cfg in platforms.items():
        if not plat_cfg.get("enabled", False):
            continue
        if platform_filter and platform_name != platform_filter:
            continue

        if platform_name == "pinterest":
            scraper = _make_pinterest(config)
        elif platform_name == "google":
            scraper = _make_google(config, pipeline.hash_store)
        else:
            logger.warning(f"Unknown platform: {platform_name}")
            continue

        logger.info(f"Platform: {platform_name.upper()}")

        for kw in keywords:
            if keyword_filter and kw != keyword_filter:
                continue
            logger.info(f"  Keyword: {kw}")

            try:
                for image_path in scraper.scrape(kw):
                    pipeline.process(image_path, kw)
                    time.sleep(pipeline.sleep_between)
            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as exc:
                logger.error(f"Error [{platform_name}] '{kw}': {exc}")
                continue

    pipeline.print_stats()
    pipeline.close()


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pose image scraper")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--platform", default="", help="pinterest | google")
    parser.add_argument("--keyword",  default="", help="scrape a single keyword")
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_logging(
        config.get("paths", {}).get("log_dir", "./logs"),
        config.get("log_level", "INFO"),
    )

    logger.info("Pose Image Scraper — Pinterest + Google → YOLO person filter")
    run(config, platform_filter=args.platform, keyword_filter=args.keyword)


if __name__ == "__main__":
    main()
