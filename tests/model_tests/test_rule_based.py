"""Tests for rule-based fallback progression."""

from pathlib import Path
import sys

import math
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rule_based import rule_based_progression


def test_drop_when_fatigued():
    res = rule_based_progression(last_weight=200, last_reps=2, last_rpe=9.5, target_reps=5)
    assert res.applied_drop is True
    assert res.suggested_weight < 200
    assert res.reason == "fatigue_or_underreps"


def test_hold_near_limit():
    res = rule_based_progression(last_weight=200, last_reps=5, last_rpe=8.7, target_reps=5)
    assert math.isclose(res.suggested_weight, 200, rel_tol=1e-3)
    assert res.reason == "hold_near_limit"


def test_easy_bump_with_cap():
    res = rule_based_progression(last_weight=100, last_reps=8, last_rpe=6.5, target_reps=5, max_increment=5)
    # With rpe < 7, expect a bump up to max_increment
    assert res.suggested_weight >= 100
    assert res.suggested_weight <= 105
    assert res.reason == "easy_session"


def test_no_history_returns_zero():
    res = rule_based_progression(last_weight=None, last_reps=None, last_rpe=None)
    assert res.suggested_weight == 0.0
    assert res.reason == "no_weight_history"


def test_rounding_to_plate():
    res = rule_based_progression(last_weight=101, last_reps=5, last_rpe=7.5, plate=2.5)
    assert res.suggested_weight % 2.5 == 0
