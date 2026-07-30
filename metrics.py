from __future__ import annotations

import math
import re

from rapidfuzz.distance import Levenshtein


def normalize_date(text: str) -> str:
    text = str(text or "MISSING").strip().lower()
    if text in {"", "null", "none", "missing"}:
        return ""
    return re.sub(r"[,.·\-\s]+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = str(text or "MISSING").strip().lower()
    return "" if text in {"", "null", "none", "missing"} else re.sub(r"\s+", " ", text)


def ned(pred: str, truth: str, *, date: bool = False) -> float:
    p = normalize_date(pred) if date else normalize_text(pred)
    t = normalize_date(truth) if date else normalize_text(truth)
    return Levenshtein.distance(p, t) / max(len(p), len(t), 1)


def aurc(risks: list[float], confidences: list[float]) -> float:
    if not risks:
        return math.nan
    order = sorted(range(len(risks)), key=lambda i: confidences[i], reverse=True)
    cumulative = 0.0
    area = 0.0
    for rank, i in enumerate(order, start=1):
        cumulative += risks[i]
        area += cumulative / rank
    return area / len(risks)
