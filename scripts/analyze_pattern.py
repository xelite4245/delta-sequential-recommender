import pandas as pd

df = pd.read_csv('data/user_inputs/user2_squat_history.csv')

# Show last 20 rows
print("Last 20 sessions:")
print(df[['weight', 'reps', 'load_delta']].tail(20).to_string())

print(f"\n\nStatistics:")
print(f"Max weight: {df['weight'].max()} lbs")
print(f"Min weight: {df['weight'].min()} lbs")
print(f"Current weight: {df.iloc[-1]['weight']} lbs")

# Detect PRs and deloads
deltas = df['load_delta'].dropna()
print(f"\nLoad delta stats:")
print(f"  Mean: {deltas.mean():.2f}")
print(f"  Std: {deltas.std():.2f}")
print(f"  Max increase: {deltas.max():.2f}")
print(f"  Max decrease: {deltas.min():.2f}")

# Show pattern
print(f"\nLoad delta pattern (last 20):")
for i, delta in enumerate(df['load_delta'].tail(20), 1):
    sign = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"  {i:2d}: {sign} {delta:+.1f}")
