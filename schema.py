from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = [
    "image_file",
    "verbatimDate",
    "verbatimDate_confidence",
    "verbatimLocality",
    "verbatimLocality_confidence",
]


def load_train(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "train.csv")
    required = {"image_file", "verbatimDate", "verbatimDate_confidence", "verbatimLocality", "verbatimLocality_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")
    return df


def load_test(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "test.csv")
    if "image_file" not in df.columns:
        raise ValueError("test.csv must contain image_file")
    return df


def validate_submission(df: pd.DataFrame, expected_ids: pd.Series | None = None) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be {SUBMISSION_COLUMNS}; got {list(df.columns)}")
    if df[SUBMISSION_COLUMNS].isna().any().any():
        raise ValueError("submission contains NaN; use MISSING for empty fields")
    if expected_ids is not None and list(df.image_file) != list(expected_ids):
        raise ValueError("submission image_file order does not match test.csv")
