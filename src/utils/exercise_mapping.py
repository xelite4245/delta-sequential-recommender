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
    # Leg exercises -> Squat
    "squat": "squat",
    "leg_press": "squat",
    "hack_squat": "squat",
    "v_squat": "squat",
    "lunges": "squat",
    "leg_extensions": "squat",
    
    # Push exercises -> Bench Press
    "bench_press": "bench_press",
    "incline_press": "bench_press",
    "dumbbell_press": "bench_press",
    "incline_db_press": "bench_press",
    "machine_press": "bench_press",
    "push_ups": "bench_press",
    
    # Vertical pull -> Lat Pulldown
    "lat_pulldown": "lat_pulldown",
    "pull_ups": "lat_pulldown",
    "chin_ups": "lat_pulldown",
    
    # Horizontal pull -> Seated Row
    "seated_row": "seated_row",
    "barbell_row": "seated_row",
    "dumbbell_row": "seated_row",
    "chest_supported_row": "seated_row",
    "machine_row": "seated_row",
    "pendulum_row": "seated_row",
}

# ===========================
# Fixed Scaling Factors
# ===========================
# These map non-core exercises to a fraction of their parent compound's load_delta
# Learned from typical strength correlations and user data patterns

SCALING_FACTORS = {
    # Squat -> Leg accessories
    "leg_press": 0.80,        # Easier than squat
    "hack_squat": 0.90,       # Similar to squat
    "v_squat": 0.85,
    "lunges": 0.60,           # Unilateral, lower absolute load
    "leg_extensions": 0.40,   # Isolation, low load progression
    
    # Bench -> Push accessories
    "incline_press": 0.85,    # Slightly harder than bench
    "dumbbell_press": 0.75,   # Dumbbells allow less load
    "machine_press": 0.80,
    "push_ups": 0.30,         # Bodyweight-based
    
    # Lat Pulldown -> Pull accessories
    "pull_ups": 0.95,         # Similar to lat pulldown
    "chin_ups": 0.95,         # Similar to lat pulldown
    
    # Seated Row -> Horizontal pull accessories
    "barbell_row": 1.05,      # Slightly more load than machine
    "dumbbell_row": 0.85,     # Dumbbells allow less load
    "chest_supported_row": 0.95,
    "machine_row": 0.90,
    "pendulum_row": 0.88,
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
