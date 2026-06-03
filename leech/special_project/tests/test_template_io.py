import pandas as pd
import pytest

from leechtemplate.coordinates import Calibration
from leechtemplate.template_io import (
    CalibrationRecord,
    load_calibration,
    load_template,
    save_calibration,
    save_template,
    validate_template,
)


def good_df():
    return pd.DataFrame(
        [
            {"name": "Retzius", "side": "L", "aspect": "ventral", "x": -0.1, "y": 0.3,
             "confidence": "high", "notes": ""},
            {"name": "Retzius", "side": "R", "aspect": "ventral", "x": 0.1, "y": 0.3,
             "confidence": "high", "notes": "ID uncertain"},
            {"name": "DE-3", "side": "M", "aspect": "ventral", "x": 0.0, "y": 0.5,
             "confidence": "medium", "notes": ""},
        ]
    )


def test_template_round_trip(tmp_path):
    path = tmp_path / "ventral.csv"
    save_template(good_df(), path)
    loaded = load_template(path)
    assert list(loaded.columns) == ["name", "side", "aspect", "x", "y", "confidence", "notes"]
    assert len(loaded) == 3
    assert loaded.loc[1, "notes"] == "ID uncertain"


def test_invalid_side_rejected():
    df = good_df()
    df.loc[0, "side"] = "X"
    with pytest.raises(ValueError):
        validate_template(df)


def test_invalid_confidence_rejected():
    df = good_df()
    df.loc[0, "confidence"] = "maybe"
    with pytest.raises(ValueError):
        validate_template(df)


def test_missing_column_rejected():
    df = good_df().drop(columns=["aspect"])
    with pytest.raises(ValueError):
        validate_template(df)


def test_calibration_round_trip(tmp_path):
    calib = Calibration((100.0, 100.0), (100.0, 300.0), (200.0, 200.0))
    record = CalibrationRecord(
        aspect="ventral", image_path="data/figures/fig.png",
        image_sha256="abc123", calibration=calib,
    )
    path = tmp_path / "ventral_calibration.json"
    save_calibration(record, path)
    back = load_calibration(path)
    assert back.aspect == "ventral"
    assert back.calibration.anterior_midline == (100.0, 100.0)
    assert back.calibration.right_ref == (200.0, 200.0)
