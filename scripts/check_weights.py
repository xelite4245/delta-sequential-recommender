import pandas as pd

df = pd.read_csv('data/user_inputs/user2_squat_history.csv')
last_weight = df.iloc[-1]['weight']
model_delta = -17.12
next_weight_from_model = last_weight + model_delta

print(f"Last weight in history: {last_weight} lbs")
print(f"ML model's load_delta: {model_delta} lbs")
print(f"Next weight from model: {last_weight} + ({model_delta}) = {next_weight_from_model} lbs")
print(f"\nRule-based suggested: 57.50 lbs")
print(f"Difference: {57.50 - next_weight_from_model:.2f} lbs")
