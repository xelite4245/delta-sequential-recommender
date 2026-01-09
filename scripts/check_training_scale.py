import pandas as pd

# Load the leg_workouts used for training
df = pd.read_csv('data/processed/PPL_data/leg_workouts.csv')

# Filter to squat only
squat = df[df['exercise_normalized'] == 'squat'].copy()
squat = squat.sort_values('date').reset_index(drop=True)

# Compute weight changes
squat['weight_change'] = squat['weight'].diff()

print("Squat data stats:")
print(f"  Rows: {len(squat)}")
print(f"  Weight range: {squat['weight'].min()} - {squat['weight'].max()}")
print(f"  Max drop: {squat['weight_change'].min()}")
print(f"  Max jump: {squat['weight_change'].max()}")

print("\nWeight changes > 30 lbs:")
big_drops = squat[squat['weight_change'] <= -30]
print(f"  Count: {len(big_drops)}")
if len(big_drops) > 0:
    print(big_drops[['date', 'weight', 'reps', 'weight_change']].head(10).to_string())

print("\nUser2 pattern (20 lbs swings) vs PPL training (30+ lbs):")
print(f"  User2 max drop: 20 lbs → is_deload = False")
print(f"  Model trained on: 30+ lbs drops → is_deload = True")
print(f"  → Model can't detect User2's cycle!")
