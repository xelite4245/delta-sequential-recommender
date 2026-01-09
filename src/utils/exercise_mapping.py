"""
Exercise-to-compound mapping and scaling factors.

Maps all exercises to one of 4 core compounds:
- Squat (leg exercises)
- Bench Press (push exercises)
- Lat Pulldown (vertical pull exercises)
- Seated Row (horizontal pull exercises)

Scaling factors are fixed, learned from data correlations.
This layer sits between predictions and application logic.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


# ===========================
# Exercise Mapping Table
# ===========================

EXERCISE_MAPPING = {
    # Push (bench press family)
    "bench_press": "bench_press",
    "close_grip_bench_press": "bench_press",
    "incline_bench_press": "bench_press",
    "low_incline_dumbbell_bench": "bench_press",
    "hex_press": "bench_press",
    "chest_fly": "bench_press",
    "cable_fly": "bench_press",
    "incline_chest_fly": "bench_press",
    "incline_cable_chest_fly": "bench_press",
    "weighted_dips": "bench_press",
    "push_up": "bench_press",
    "push_ups": "bench_press",
    "overhead_press": "bench_press",
    "seated_overhead_press": "bench_press",
    "shoulder_press_machine": "bench_press",
    "seated_shoulder_press_machine": "bench_press",
    "lateral_raise": "bench_press",
    "upright_row": "bench_press",
    "lying_skullcrusher": "bench_press",
    "triceps_extension": "bench_press",
    "triceps_pushdown": "bench_press",

    # Vertical pull (lat pulldown family)
    "lat_pulldown": "lat_pulldown",
    "lat_pulldown_wide": "lat_pulldown",
    "lat_pulldown_close": "lat_pulldown",
    "lat_pulldown_underhand_close": "lat_pulldown",
    "lat_pull_in": "lat_pulldown",
    "chin_up": "lat_pulldown",
    "chin_ups": "lat_pulldown",
    "neutral_chin": "lat_pulldown",
    "pull_ups": "lat_pulldown",

    # Horizontal pull (seated row family)
    "seated_row": "seated_row",
    "seated_cable_row": "seated_row",
    "hammer_seated_row": "seated_row",
    "hammer_row_wide": "seated_row",
    "bent_over_row": "seated_row",
    "bent_over_one_arm_row": "seated_row",
    "incline_row": "seated_row",
    "face_pull": "seated_row",
    "reverse_fly": "seated_row",
    "shrug": "seated_row",
    "barbell_curl": "seated_row",
    "ez_bar_curl": "seated_row",
    "dumbbell_curl": "seated_row",
    "incline_curl": "seated_row",
    "hammer_curl": "seated_row",
    # Legacy names kept for compatibility
    "barbell_row": "seated_row",
    "dumbbell_row": "seated_row",
    "chest_supported_row": "seated_row",
    "machine_row": "seated_row",
    "pendulum_row": "seated_row",

    # Lower body (squat family)
    "squat": "squat",
    "high_bar_squat": "squat",
    "front_squat": "squat",
    "hack_squat": "squat",
    "leg_press": "squat",
    "seated_leg_press": "squat",
    "leg_extension": "squat",
    "leg_extensions": "squat",
    "leg_outward_fly": "squat",
    "deadlift": "squat",
    "trap_bar_deadlift": "squat",
    "sumo_deadlift": "squat",
    "romanian_deadlift": "squat",
    "rack_pull_1_pin": "squat",
    "rack_pull_2_pin": "squat",
    "good_morning": "squat",
    "glute_extension": "squat",
    "seated_leg_curl": "squat",
    "lying_leg_curl": "squat",
    "calf_press_seated_leg_press": "squat",
}

# ===========================
# Fixed Scaling Factors
# ===========================
# These map non-core exercises to a fraction of their parent compound's load_delta
# Learned from typical strength correlations and user data patterns

SCALING_FACTORS = {
    # Bench / chest / shoulders / triceps (relative to bench_press)
    "bench_press": 1.00,
    "close_grip_bench_press": 0.95,
    "incline_bench_press": 0.85,
    "low_incline_dumbbell_bench": 0.75,
    "hex_press": 0.70,
    "chest_fly": 0.40,
    "cable_fly": 0.45,
    "incline_chest_fly": 0.40,
    "incline_cable_chest_fly": 0.45,
    "weighted_dips": 0.90,
    "push_up": 0.30,
    "push_ups": 0.30,
    "overhead_press": 0.65,
    "seated_overhead_press": 0.60,
    "shoulder_press_machine": 0.55,
    "seated_shoulder_press_machine": 0.50,
    "lateral_raise": 0.25,
    "upright_row": 0.60,
    "lying_skullcrusher": 0.45,
    "triceps_extension": 0.40,
    "triceps_pushdown": 0.35,

    # Vertical pull (relative to lat_pulldown)
    "lat_pulldown": 1.00,
    "lat_pulldown_wide": 0.95,
    "lat_pulldown_close": 1.05,
    "lat_pulldown_underhand_close": 1.10,
    "lat_pull_in": 0.85,
    "chin_up": 0.90,
    "chin_ups": 0.90,
    "neutral_chin": 0.85,
    "pull_ups": 0.95,

    # Horizontal pull / upper back / biceps (relative to seated_row)
    "seated_row": 1.00,
    "seated_cable_row": 0.95,
    "hammer_seated_row": 0.90,
    "hammer_row_wide": 0.85,
    "bent_over_row": 1.05,
    "bent_over_one_arm_row": 0.85,
    "incline_row": 0.95,
    "face_pull": 0.35,
    "reverse_fly": 0.30,
    "shrug": 0.75,
    "barbell_curl": 0.40,
    "ez_bar_curl": 0.38,
    "dumbbell_curl": 0.35,
    "incline_curl": 0.30,
    "hammer_curl": 0.36,
    # Legacy row names
    "barbell_row": 1.05,
    "dumbbell_row": 0.85,
    "chest_supported_row": 0.95,
    "machine_row": 0.90,
    "pendulum_row": 0.88,

    # Squat family (lower body)
    "squat": 1.00,
    "high_bar_squat": 0.95,
    "front_squat": 0.85,
    "hack_squat": 0.90,
    "leg_press": 0.80,
    "seated_leg_press": 0.75,
    "leg_extension": 0.40,
    "leg_extensions": 0.40,
    "leg_outward_fly": 0.30,
    "deadlift": 1.20,
    "trap_bar_deadlift": 1.25,
    "sumo_deadlift": 1.25,
    "romanian_deadlift": 0.90,
    "rack_pull_1_pin": 1.30,
    "rack_pull_2_pin": 1.40,
    "good_morning": 0.60,
    "glute_extension": 0.45,
    "seated_leg_curl": 0.40,
    "lying_leg_curl": 0.45,
    "calf_press_seated_leg_press": 0.60,
}


# ===========================
# Lookup Functions
# ===========================

def get_parent_compound(exercise: str) -> str:
    """
    Get parent compound for an exercise.
    
    Args:
        exercise: Normalized exercise name
        
    Returns:
        Parent compound name or exercise itself if not mapped
    """
    normalized = exercise.lower().strip()
    return EXERCISE_MAPPING.get(normalized, normalized)


def get_scaling_factor(exercise: str) -> float:
    """
    Get scaling factor for an exercise relative to its parent compound.
    
    Args:
        exercise: Normalized exercise name
        
    Returns:
        Scaling factor (default 1.0 if not in table)
    """
    normalized = exercise.lower().strip()
    
    # If it's a core compound, scaling factor is 1.0
    if normalized in ["squat", "bench_press", "lat_pulldown", "seated_row"]:
        return 1.0
    
    return SCALING_FACTORS.get(normalized, 1.0)


def predict_exercise_delta(
    parent_compound_delta: float,
    exercise: str
) -> float:
    """
    Predict load delta for any exercise using its parent compound model.
    
    Args:
        parent_compound_delta: Prediction from parent compound model
        exercise: Target exercise name
        
    Returns:
        Scaled prediction for the exercise
    """
    scaling_factor = get_scaling_factor(exercise)
    return parent_compound_delta * scaling_factor


def compute_scaling_factors_from_data(
    df: pd.DataFrame,
    parent_compound: str
) -> Dict[str, float]:
    """
    Optionally: compute scaling factors from actual data correlations.
    (For future refinement; currently using fixed factors.)
    
    Args:
        df: Dataframe with exercise, weight, date columns
        parent_compound: Name of parent compound (e.g., "squat")
        
    Returns:
        Dict mapping exercise -> scaling_factor
    """
    # Group by exercise, compute average load_delta
    # Then ratio child_delta / parent_delta
    
    # For now, return empty dict (use fixed factors above)
    return {}


# ===========================
# Utilities
# ===========================

def categorize_exercises(exercises: list) -> Dict[str, list]:
    """
    Categorize a list of exercises by their parent compound.
    
    Args:
        exercises: List of normalized exercise names
        
    Returns:
        Dict mapping compound -> list of exercises in that category
    """
    categorized = {
        "squat": [],
        "bench_press": [],
        "lat_pulldown": [],
        "seated_row": [],
    }
    
    for exercise in exercises:
        parent = get_parent_compound(exercise)
        if parent in categorized:
            categorized[parent].append(exercise)
        else:
            # Unknown parent; put in "other"
            if "other" not in categorized:
                categorized["other"] = []
            categorized["other"].append(exercise)
    
    return categorized


def print_exercise_mapping():
    """Pretty-print the exercise mapping table."""
    print("=" * 70)
    print("EXERCISE MAPPING (Exercise -> Parent Compound)")
    print("=" * 70)
    
    compounds = {}
    for exercise, parent in EXERCISE_MAPPING.items():
        if parent not in compounds:
            compounds[parent] = []
        compounds[parent].append(exercise)
    
    for parent in ["squat", "bench_press", "lat_pulldown", "seated_row"]:
        if parent in compounds:
            print(f"\n{parent.upper()}:")
            for exercise in sorted(compounds[parent]):
                scaling = SCALING_FACTORS.get(exercise, 1.0)
                print(f"  {exercise:<30} (scaling: {scaling:.2f})")
