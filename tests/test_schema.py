import pandas as pd

from museumscat.src.schema import SUBMISSION_COLUMNS, validate_submission


def test_submission_schema():
    df = pd.DataFrame([{"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 0.2, "verbatimLocality": "MISSING", "verbatimLocality_confidence": 0.2}])
    validate_submission(df, expected_ids=pd.Series(["a.jpeg"]))
    assert list(df.columns) == SUBMISSION_COLUMNS
