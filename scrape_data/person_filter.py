"""
Person filter using YOLOv11 (ultralytics).

Rule: the largest detected person bbox must cover >= min_area_ratio of the
full image area (default 1/3 = 0.333).  Images with no person or a person
that is too small are rejected.

Model is loaded once and cached as a module-level singleton to avoid
re-loading for every image.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from ultralytics import YOLO

_model_cache: dict[str, YOLO] = {}


def _get_model(model_path: str) -> YOLO:
    if model_path not in _model_cache:
        logger.info(f"[person_filter] loading YOLO model: {model_path}")
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def check_person(
    image: np.ndarray,
    model_path: str = "yolo11x.pt",
    confidence: float = 0.4,
    min_area_ratio: float = 1 / 3,
) -> tuple[bool, dict]:
    """
    Returns (passed, details).

    passed=True  → largest person bbox >= min_area_ratio of image area
    passed=False → no person detected, or person too small
    """
    model = _get_model(model_path)
    results = model(image, conf=confidence, verbose=False, workers=0)

    h, w = image.shape[:2]
    image_area = h * w

    persons = []
    if results:
        r = results[0]
        boxes = r.boxes
        if boxes is not None:
            for i, cls in enumerate(boxes.cls):
                if int(cls) != 0:   # class 0 = person in COCO
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                area = float((x2 - x1) * (y2 - y1))
                persons.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "area": area,
                    "conf": float(boxes.conf[i]),
                })

    if not persons:
        return False, {"reason": "no_person", "person_count": 0}

    persons.sort(key=lambda p: p["area"], reverse=True)
    largest = persons[0]
    ratio = largest["area"] / image_area

    if ratio < min_area_ratio:
        return False, {
            "reason": "person_too_small",
            "person_count": len(persons),
            "largest_ratio": round(ratio, 4),
            "required_ratio": min_area_ratio,
        }

    return True, {
        "person_count": len(persons),
        "largest_ratio": round(ratio, 4),
        "bbox": largest["bbox"],
    }
