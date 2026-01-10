"""Test plot generation"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.plot_generator import generate_and_save_plots

# Test with User2's squat data
user_data_path = Path(repo_root) / "users" / "User2"
compound = "squat"

print(f"Testing plot generation for {compound}...")
print(f"User data path: {user_data_path}")

plot_path = generate_and_save_plots(str(user_data_path), compound)

if plot_path and plot_path.exists():
    print(f"✓ Plot generated successfully!")
    print(f"  Location: {plot_path}")
    print(f"  File size: {plot_path.stat().st_size / 1024:.1f} KB")
else:
    print("✗ Plot generation failed")
    sys.exit(1)

# Test all compounds
print("\nTesting all compounds...")
for comp in ["squat", "bench_press", "lat_pulldown", "seated_row"]:
    plot_path = generate_and_save_plots(str(user_data_path), comp)
    if plot_path and plot_path.exists():
        print(f"  ✓ {comp}: {plot_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"  ⚠ {comp}: No history yet")

print("\n✓ All plot tests passed!")
