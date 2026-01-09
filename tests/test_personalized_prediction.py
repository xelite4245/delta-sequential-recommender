"""Unit tests for prediction path with affine per-user calibration."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Allow running tests without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from personalized_prediction import (
    CalibrationConfig,
    predict_with_user_calibration,
)
from models.base_model import BaseModel
from utils.user_personalization import PersonalizationRegistry


def _train_simple_model() -> BaseModel:
    model = BaseModel(estimator=LinearRegression(), model_name="test_model")
    X_train = pd.DataFrame({"feature": np.arange(50, dtype=float)})
    y_train = X_train["feature"] * 1.0  # baseline slope 1
    model.fit(X_train, y_train)
    return model


def _make_history(n: int) -> pd.DataFrame:
    feature = np.arange(1, n + 1, dtype=float)
    # True relation intentionally different from model to force calibration
    load_delta = (feature * 2.0) + 5.0
    return pd.DataFrame({"feature": feature, "load_delta": load_delta})


def test_calibration_triggers_and_applies() -> None:
    model = _train_simple_model()
    history = _make_history(20)
    future = pd.DataFrame({"feature": np.array([21.0, 22.0])})

    cfg = CalibrationConfig(min_samples=8, refit_every=5, calibration_window=32)

    with TemporaryDirectory() as tmp:
        registry = PersonalizationRegistry(base_dir=Path(tmp))
        raw, adjusted, ab = predict_with_user_calibration(
            model=model,
            user_id="u1",
            compound="squat",
            future_df=future,
            registry=registry,
            history=history,
            config=cfg,
        )

        assert ab is not None, "Calibration should run with enough samples"
        a, b = ab
        # Gain is clamped to the configured bounds (default upper=1.4)
        assert 1.2 <= a <= cfg.gain_bounds[1]
        assert 0.0 < b < 10.0
        # Raw predictions use slope ~1, adjusted should follow ~2x + 5
        expected = (raw * a) + b
        assert np.allclose(adjusted, expected)

        up = registry.get_or_create("u1")
        meta = up.calibration_meta["squat"]
        assert meta["runs"] == 1
        assert meta["last_calibrated_size"] == len(history)


def test_refit_respects_cadence() -> None:
    model = _train_simple_model()
    history = _make_history(20)
    future = pd.DataFrame({"feature": np.array([23.0, 24.0])})
    cfg = CalibrationConfig(min_samples=8, refit_every=5, calibration_window=32)

    with TemporaryDirectory() as tmp:
        registry = PersonalizationRegistry(base_dir=Path(tmp))
        # First call calibrates
        _, _, ab1 = predict_with_user_calibration(
            model=model,
            user_id="u2",
            compound="squat",
            future_df=future,
            registry=registry,
            history=history,
            config=cfg,
        )
        assert ab1 is not None

        # Append only two more samples (< refit_every); should not refit
        history_extra = _make_history(22)
        _, adjusted, ab2 = predict_with_user_calibration(
            model=model,
            user_id="u2",
            compound="squat",
            future_df=future,
            registry=registry,
            history=history_extra,
            config=cfg,
        )
        assert ab2 is None, "Calibration should skip when cadence not met"

        up = registry.get_or_create("u2")
        assert up.calibration_meta["squat"]["runs"] == 1
        # Adjusted predictions should still be applied using stored (a, b)
        a, b = up.scaling_factors["squat"], up.baseline_offsets["squat"]
        raw_again = model.predict(future)
        expected = (raw_again * a) + b
        assert np.allclose(adjusted, expected)


def test_calibration_applies_adjustment() -> None:
    model = _train_simple_model()
    # History with slope 2 and offset 5
    history = _make_history(12)
    future = pd.DataFrame({"feature": np.array([30.0, 40.0])})
    cfg = CalibrationConfig(min_samples=8, refit_every=5, calibration_window=32)

    with TemporaryDirectory() as tmp:
        registry = PersonalizationRegistry(base_dir=Path(tmp))
        raw, adjusted, ab = predict_with_user_calibration(
            model=model,
            user_id="u3",
            compound="squat",
            future_df=future,
            registry=registry,
            history=history,
            config=cfg,
        )

        assert ab is not None
        a, b = ab
        # Model predicts ~x; calibrated should approximate 2x+5 pattern
        expected = (raw * a) + b
        assert np.allclose(adjusted, expected)
        # Persisted calibration should match
        up = registry.get_or_create("u3")
        assert math.isclose(up.scaling_factors["squat"], a)
        assert math.isclose(up.baseline_offsets["squat"], b)
