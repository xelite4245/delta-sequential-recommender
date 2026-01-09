"""Rule-based deterministic progression fallback.

Used for cold-start, sanity fallback, or when a model output is missing.
Produces a suggested top-set weight adjustment based on last session reps/RPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class RuleBasedSuggestion:
    suggested_weight: float
    reason: str
    applied_drop: bool
    applied_cap: bool


def _round_to_plate(weight: float, plate: float = 2.5) -> float:
    """Round to the nearest plate increment (default 2.5 lbs)."""
    if plate <= 0:
        return weight
    return round(weight / plate) * plate


def rule_based_progression(
    *,
    last_weight: Optional[float],
    last_reps: Optional[float],
    last_rpe: Optional[float],
    target_reps: int = 5,
    max_increment: float = 5.0,
    drop_fraction: float = 0.05,
    plate: float = 2.5,
) -> RuleBasedSuggestion:
    """
    Simple deterministic heuristic:
    - If prior set was very hard (RPE > 9 or reps << target), drop by `drop_fraction`.
    - If solid but near limit (RPE >= 8.5), hold weight.
    - If comfortably hard (7 <= RPE < 8.5), small bump (max 2.5 lbs or fraction of last).
    - If easy (RPE < 7), larger bump up to `max_increment`.
    Falls back to no-change if inputs are missing.
    """
    if last_weight is None or math.isnan(last_weight):
        return RuleBasedSuggestion(suggested_weight=0.0, reason="no_weight_history", applied_drop=False, applied_cap=False)

    reps_gap = None if last_reps is None or math.isnan(last_reps) else (target_reps - last_reps)
    rpe = last_rpe if last_rpe is not None and not math.isnan(last_rpe) else None

    applied_drop = False
    applied_cap = False

    # Base suggestions
    if (rpe is not None and rpe > 9) or (reps_gap is not None and reps_gap >= 2):
        delta = -last_weight * drop_fraction
        applied_drop = True
        reason = "fatigue_or_underreps"
    elif rpe is not None and rpe >= 8.5:
        delta = 0.0
        reason = "hold_near_limit"
    elif rpe is not None and rpe < 7:
        delta = min(max_increment, last_weight * 0.05)
        reason = "easy_session"
    else:
        delta = min(max_increment / 2.0, last_weight * 0.025)
        reason = "steady_progress"

    suggested = last_weight + delta
    suggested = _round_to_plate(suggested, plate=plate)

    # Cap extreme suggestions
    if suggested > last_weight + max_increment:
        suggested = last_weight + max_increment
        suggested = _round_to_plate(suggested, plate=plate)
        applied_cap = True
    if suggested < last_weight * (1 - 2 * drop_fraction):
        suggested = last_weight * (1 - 2 * drop_fraction)
        suggested = _round_to_plate(suggested, plate=plate)
        applied_cap = True

    return RuleBasedSuggestion(
        suggested_weight=float(suggested),
        reason=reason,
        applied_drop=applied_drop,
        applied_cap=applied_cap,
    )
