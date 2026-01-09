import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.compound_models import add_periodization_features

# Load PPL squat data and apply same cycle detection
df = pd.read_csv('data/processed/PPL_data/leg_workouts.csv')
squat = df[df['exercise_normalized'] == 'squat'].copy()
squat = squat.sort_values('date').reset_index(drop=True)

# Filter to top sets
if 'set_order' in squat.columns:
    squat = squat[squat['set_order'] == 1]

# Compute load_delta and apply periodization
squat['load_delta'] = squat['weight'].diff().shift(-1)
squat = add_periodization_features(squat)

# Look at what happens after 3 weeks in cycle at 70-75% of max
near_max_3weeks = squat[
    (squat['weeks_in_cycle'] == 3) & 
    (squat['percent_of_max'] >= 0.70) & 
    (squat['percent_of_max'] <= 0.75)
]

print(f"Training pattern: weeks_in_cycle=3, percent_of_max=70-75%")
print(f"  Count: {len(near_max_3weeks)}")
if len(near_max_3weeks) > 0:
    print(f"  Average load_delta: {near_max_3weeks['load_delta'].mean():.2f}")
    print(f"  Median load_delta: {near_max_3weeks['load_delta'].median():.2f}")
    print(f"  Range: {near_max_3weeks['load_delta'].min():.2f} to {near_max_3weeks['load_delta'].max():.2f}")
    
    print("\n  Sample rows:")
    print(near_max_3weeks[['weight', 'reps', 'load_delta', 'cycle_number', 'weeks_in_cycle', 'percent_of_max']].head(15).to_string())
