from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOCR:
    date: str
    locality: str
    date_confidence: float
    locality_confidence: float
    raw: str
    parse_method: str


def _clean(value: object) -> str:
    if value is None:
        return "MISSING"
    value = str(value).strip()
    if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
        return "MISSING"
    return value


def _extract_json(raw: str) -> dict | None:
    candidates = [raw.strip()] + re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_ocr(raw: str) -> ParsedOCR:
    raw = raw or ""
    obj = _extract_json(raw)
    if obj is not None:
        date = _clean(obj.get("verbatimDate", obj.get("date")))
        locality = _clean(obj.get("verbatimLocality", obj.get("locality")))
        base = 0.85 if date != "MISSING" and locality != "MISSING" else 0.55
        return ParsedOCR(date, locality, base, base, raw, "json")
    date_match = re.search(r"(?:verbatimDate|date)\s*[:=]\s*(.+)", raw, flags=re.I)
    locality_match = re.search(r"(?:verbatimLocality|locality)\s*[:=]\s*(.+)", raw, flags=re.I)
    date = _clean(date_match.group(1).splitlines()[0]) if date_match else "MISSING"
    locality = _clean(locality_match.group(1).splitlines()[0]) if locality_match else "MISSING"
    base = 0.45 if date != "MISSING" or locality != "MISSING" else 0.2
    method = "regex" if date != "MISSING" or locality != "MISSING" else "fallback"
    return ParsedOCR(date, locality, base, base, raw, method)
