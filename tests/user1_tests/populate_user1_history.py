#!/usr/bin/env python3
"""
Populate user1's workout history with validation data from the baseline.

This creates realistic logged workouts for user1 by taking the validation
portion of the baseline dataset and formatting it as workout history.
"""
import sys
import pandas as pd
from pathlib import Path

# Add repo root to path (go up 3 levels from tests/user1_tests/)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.exercise_mapping import get_parent_compound


def normalize_exercise_name(exercise: str) -> str:
    """Normalize exercise name from CSV format to mapping format"""
    # Convert to lowercase and replace spaces/hyphens with underscores
    normalized = exercise.lower().strip()
    normalized = normalized.replace(' ', '_').replace('-', '_')
    
    # Remove parenthetical content (e.g., "Bench Press (Barbell)" -> "bench_press")
    normalized = normalized.split('(')[0].strip().replace(' ', '_')
    
    return normalized


def populate_user1_history():
    """Populate user1's history with baseline validation data"""
    
    print("=" * 70)
    print("POPULATING USER1 WORKOUT HISTORY")
    print("=" * 70)
    
    try:
        # Load baseline data
        baseline_csv = repo_root / "data" / "baseline" / "strong_4krows_baseline_data.csv"
        # Print a repo-relative path to avoid exposing absolute user directories
        try:
            rel = baseline_csv.relative_to(repo_root)
            display_path = rel.as_posix()
        except Exception:
            display_path = baseline_csv.name
        print(f"\n1. Loading baseline data from: {display_path}")
        
        df = pd.read_csv(baseline_csv)
        print(f"   ✓ Loaded {len(df)} rows")
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Date': 'date',
            'Exercise Name': 'exercise_name',
            'Weight': 'weight',
            'Reps': 'reps',
            'Set Order': 'set_order',
            'RPE': 'rpe'
        })
        
        # Clean up data
        df['date'] = pd.to_datetime(df['date'])
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
        df['set_order'] = pd.to_numeric(df['set_order'], errors='coerce')
        df['rpe'] = pd.to_numeric(df['rpe'], errors='coerce')
        
        # Filter to top sets only
        df = df[df['set_order'] == 1].copy()
        df = df.dropna(subset=['weight', 'reps', 'exercise_name'])
        print(f"   ✓ Filtered to {len(df)} top sets")
        
        # Normalize exercise names and map to compounds
        df['exercise_normalized'] = df['exercise_name'].apply(normalize_exercise_name)
        df['compound'] = df['exercise_normalized'].apply(get_parent_compound)
        
        # Remove rows where compound mapping failed (compound == normalized name means no mapping found)
        valid_compounds = ['squat', 'bench_press', 'lat_pulldown', 'seated_row']
        df = df[df['compound'].isin(valid_compounds)].copy()
        print(f"   ✓ Mapped to compounds, {len(df)} rows with valid exercises")
        
        # Show some debug info
        print(f"   ℹ  Compound distribution:")
        for compound in valid_compounds:
            count = len(df[df['compound'] == compound])
            print(f"      {compound}: {count} sets")
        
        # Process each compound separately
        user_path = repo_root / "users" / "user1"
        
        for compound in valid_compounds:
            print(f"\n2. Processing {compound}...")
            
            # Filter to compound
            df_compound = df[df['compound'] == compound].copy()
            df_compound = df_compound.sort_values('date').reset_index(drop=True)
            
            if len(df_compound) < 14:
                print(f"   ⚠  Only {len(df_compound)} rows for {compound}. Using all available.")
                df_use = df_compound.copy()
            else:
                # Take the last 14 entries (validation/test portion)
                df_use = df_compound.iloc[-14:].copy()
            
            if len(df_use) == 0:
                print(f"   ⚠  No data for {compound}. Skipping.")
                continue
            
            # Compute load_delta
            df_use = df_use.reset_index(drop=True)
            df_use['load_delta'] = df_use['weight'].diff().fillna(0).astype(float)
            
            # Select columns needed for history CSV
            df_output = df_use[['weight', 'reps', 'rpe', 'load_delta']].copy()
            df_output = df_output.reset_index(drop=True)
            
            # Write to user history CSV
            history_csv = user_path / f"user1_{compound}_history.csv"
            df_output.to_csv(history_csv, index=False)
            
            print(f"   ✓ Wrote {len(df_output)} sessions to {history_csv.name}")
            print(f"     Weight range: {df_output['weight'].min():.1f} - {df_output['weight'].max():.1f} lbs")
            if df_output['rpe'].notna().any():
                print(f"     Average RPE: {df_output['rpe'].mean():.1f}")
            print(f"     Load deltas: {df_output['load_delta'].min():.1f} to {df_output['load_delta'].max():.1f} lbs")
        
        # Summary
        print("\n" + "=" * 70)
        print("✓ USER1 HISTORY POPULATED")
        print("=" * 70)
        print("\nuser1 now has realistic workout history and can receive ML predictions!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = populate_user1_history()
    sys.exit(0 if success else 1)
