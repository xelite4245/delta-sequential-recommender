import pandas as pd
import numpy as np

# The current User2 prediction was:
# raw: -32.07, adjusted: -18.01, fitted: (a, b) = unknown

# Let's see the calibration that was saved
import json
from pathlib import Path

pers_path = Path("users/User2/personalization.json")
if pers_path.exists():
    with open(pers_path) as f:
        pers = json.load(f)
    
    print("User2 Personalization (after retraining):")
    print(f"  squat a (gain): {pers['scaling_factors']['squat']}")
    print(f"  squat b (offset): {pers['baseline_offsets']['squat']}")
    print(f"  Last calibration size: {pers['calibration_meta']['squat']['last_calibrated_size']}")
    print(f"  Calibration runs: {pers['calibration_meta']['squat']['runs']}")
    
    print("\nInterpretation:")
    print(f"  Adjusted = {pers['scaling_factors']['squat']:.2f} × raw + {pers['baseline_offsets']['squat']:.2f}")
    print(f"  -32.07 × {pers['scaling_factors']['squat']:.2f} + {pers['baseline_offsets']['squat']:.2f} = {-32.07 * pers['scaling_factors']['squat'] + pers['baseline_offsets']['squat']:.2f}")
else:
    print("No personalization found")
