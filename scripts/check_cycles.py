import pandas as pd
import numpy as np

# Simulate what the model sees
df = pd.read_csv('data/user_inputs/user2_squat_history.csv')

# Apply the new percentage-based deload detection
df_copy = df.copy()
df_copy = df_copy.sort_values('date').reset_index(drop=True)

prev_weight = df_copy['weight'].shift(1)
pct_change = (prev_weight - df_copy['weight']) / prev_weight
df_copy['is_deload'] = pct_change >= 0.15
df_copy['cycle_number'] = df_copy['is_deload'].cumsum()
df_copy['weeks_in_cycle'] = df_copy.groupby('cycle_number').cumcount() + 1
df_copy['max_weight_so_far'] = df_copy['weight'].expanding().max()
df_copy['percent_of_max'] = df_copy['weight'] / df_copy['max_weight_so_far']

print("Last 20 rows with cycle features:")
print(df_copy[['weight', 'reps', 'load_delta', 'is_deload', 'cycle_number', 'weeks_in_cycle', 'percent_of_max']].tail(20).to_string())

print(f"\n\nDeload detections (15% drop):")
deloads = df_copy[df_copy['is_deload']]
print(f"  Count: {len(deloads)}")
if len(deloads) > 0:
    print(deloads[['weight', 'load_delta', 'cycle_number', 'weeks_in_cycle']].head(10).to_string())

print(f"\n\nCurrent state:")
print(f"  Cycle #: {df_copy.iloc[-1]['cycle_number']}")
print(f"  Weeks in cycle: {df_copy.iloc[-1]['weeks_in_cycle']}")
print(f"  % of max: {df_copy.iloc[-1]['percent_of_max']:.2%}")
print(f"  Next load_delta in training: {df_copy.iloc[-1]['load_delta']}")
