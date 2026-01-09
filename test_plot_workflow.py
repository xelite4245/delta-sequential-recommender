"""Enhanced workflow test that verifies plot generation"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from src import auth, session_logger
from src.plot_generator import generate_and_save_plots

print("="*60)
print("ENHANCED WORKFLOW TEST (with plot generation)")
print("="*60)

# Test data
username = "User2"
compound = "squat"
user_data_path = Path(repo_root) / "users" / "User2"

print("\n1. Testing plot generation...")
plot_path = generate_and_save_plots(str(user_data_path), compound)
if plot_path and plot_path.exists():
    file_size = plot_path.stat().st_size / 1024
    print(f"   ✓ Plot generated: {plot_path}")
    print(f"   ✓ File size: {file_size:.1f} KB")
else:
    print("   ✗ Plot generation failed")
    sys.exit(1)

print("\n2. Simulating session log...")
session_logger.log_session(
    user_id=1,
    user_data_path=str(user_data_path),
    compound=compound,
    weight=200.0,
    reps=5,
    rpe=7.5,
    deviation_reason="normal",
    recommended_weight=202.5,
    prediction_source="rule_based"
)
print("   ✓ Session logged")

print("\n3. Regenerating plots after session...")
plot_path = generate_and_save_plots(str(user_data_path), compound)
if plot_path and plot_path.exists():
    file_size = plot_path.stat().st_size / 1024
    print(f"   ✓ Plot updated: {plot_path}")
    print(f"   ✓ File size: {file_size:.1f} KB")
else:
    print("   ✗ Plot update failed")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Plot Generation Working!")
print("="*60)
print("\nThe app will now:")
print("  1. Log each session")
print("  2. Regenerate plots automatically")
print("  3. Allow users to view plots via CLI menu")
print("\nRun: python run_app.py")
