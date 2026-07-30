from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .parse_ocr import parse_ocr
from .prompt import BASE_PROMPT
from .schema import SUBMISSION_COLUMNS, validate_submission


def infer_image(model, tokenizer, image_path: Path, output_dir: Path) -> str:
    before = set(output_dir.glob("**/*"))
    result = model.infer(
        tokenizer,
        prompt=BASE_PROMPT,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=1024,
        crop_mode=False,
        max_length=4096,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )
    if isinstance(result, str) and result.strip():
        return result
    after = [p for p in output_dir.glob("**/*") if p not in before and p.is_file()]
    for path in sorted(after, key=lambda p: p.stat().st_mtime, reverse=True):
        if path.suffix.lower() in {".txt", ".md", ".json"}:
            return path.read_text(errors="replace")
    return ""


def run_baseline(model, tokenizer, data_dir: str | Path, artifact_dir: str | Path, limit: int | None = None) -> Path:
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(data_dir / "test.csv")
    if limit:
        test = test.head(limit)
    rows, raw_records = [], []
    for row in test.itertuples(index=False):
        started = time.time()
        raw_dir = artifact_dir / "raw" / Path(row.image_file).stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw = infer_image(model, tokenizer, data_dir / "images" / row.image_file, raw_dir)
        parsed = parse_ocr(raw)
        rows.append({
            "image_file": row.image_file,
            "verbatimDate": parsed.date,
            "verbatimDate_confidence": parsed.date_confidence,
            "verbatimLocality": parsed.locality,
            "verbatimLocality_confidence": parsed.locality_confidence,
        })
        raw_records.append({"image_file": row.image_file, "raw": raw, "parse_method": parsed.parse_method, "latency_s": time.time() - started})
    submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, expected_ids=test.image_file)
    out = artifact_dir / "submission_unlimited_ocr.csv"
    submission.to_csv(out, index=False)
    (artifact_dir / "raw_outputs.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in raw_records) + "\n")
    return out
