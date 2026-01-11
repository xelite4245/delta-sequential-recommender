#!/usr/bin/env python3
"""
Setup script to create a demo user (user1) with pre-trained models on baseline data.
"""
import sys
import sqlite3
import json
import shutil
import pandas as pd
from pathlib import Path

# Add repo root to path (go up 3 levels from tests/user1_tests/)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.models.compound_models import CompoundProgressionModel


def create_demo_user():
    """Create user1 with demo123 password and pre-trained models"""
    
    print("=" * 70)
    print("SETTING UP DEMO USER (user1)")
    print("=" * 70)
    
    # Step 1: Register user in auth database
    print("\n1. Registering user1 in auth database...")
    try:
        DB_PATH = repo_root / "data" / "auth" / "app_users.db"
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Check if user exists
        cur.execute("SELECT user_id FROM users WHERE username = ?", ("user1",))
        if cur.fetchone() is not None:
            print("   ℹ  User user1 already exists. Removing old entry...")
            cur.execute("DELETE FROM users WHERE username = ?", ("user1",))
            conn.commit()
        
        # Create user directory
        user_data_path = repo_root / "users" / "user1"
        user_data_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ User directory created: {user_data_path}")
        
        # Create empty CSV files for each compound
        compounds = ["squat", "bench_press", "lat_pulldown", "seated_row"]
        for compound in compounds:
            csv_path = user_data_path / f"user1_{compound}_history.csv"
            csv_path.write_text("weight,reps,rpe,load_delta\n")
        print(f"   ✓ Created {len(compounds)} history CSV files")
        
        # Create personalization.json
        personalization = {
            "scaling_factors": {c: 1.0 for c in compounds},
            "baseline_offsets": {c: 0.0 for c in compounds},
            "last_calibration_n": {c: 0 for c in compounds},
            "calibration_runs": {c: 0 for c in compounds},
        }
        pers_path = user_data_path / "personalization.json"
        pers_path.write_text(json.dumps(personalization, indent=2))
        print(f"   ✓ Created personalization.json")
        
        # Insert user into database
        cur.execute("""
            INSERT INTO users (username, password, user_data_path)
            VALUES (?, ?, ?)
        """, ("user1", "demo123", str(user_data_path)))
        
        user_id = cur.lastrowid
        conn.commit()
        conn.close()
        
        print(f"   ✓ User registered with user_id={user_id}")
        
    except Exception as e:
        print(f"   ✗ Error registering user: {e}")
        return False
    
    # Step 2: Train models on baseline data
    print("\n2. Training models on baseline data...")
    try:
        baseline_dir = repo_root / "data" / "baseline"
        baseline_csv = baseline_dir / "strong_4krows_baseline_data.csv"
        
        if not baseline_csv.exists():
            print(f"   ✗ Baseline CSV not found: {baseline_csv}")
            return False
        
        print(f"   ℹ  Loading baseline data from: {baseline_csv}")
        
        # Create models directory if it doesn't exist
        models_dir = repo_root / "models" / "compounds"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load baseline data
        df = pd.read_csv(baseline_csv)
        print(f"   ℹ  Loaded {len(df)} rows from baseline")
        
        # Train models for each compound
        compounds_to_train = {
            "squat": ("Squat", "leg_exercises"),
            "bench_press": ("Bench Press", "push_exercises"),
            "lat_pulldown": ("Lat Pulldown", "vertical_pull"),
            "seated_row": ("Seated Row", "horizontal_pull")
        }
        
        for compound, (display_name, exercise_filter) in compounds_to_train.items():
            print(f"\n   Training {compound} model...")
            try:
                model = CompoundProgressionModel(name=display_name)
                
                # Filter to a compound subset for training
                # This is a simple approach - filter by exercise keywords
                if compound == "squat":
                    df_filtered = df[df['Exercise Name'].str.contains('Squat|Leg Press|Deadlift|Leg Extension', case=False, na=False)].copy()
                elif compound == "bench_press":
                    df_filtered = df[df['Exercise Name'].str.contains('Bench|Chest|Triceps|Overhead Press', case=False, na=False)].copy()
                elif compound == "lat_pulldown":
                    df_filtered = df[df['Exercise Name'].str.contains('Lat|Pull|Chin|Pulldown', case=False, na=False)].copy()
                else:  # seated_row
                    df_filtered = df[df['Exercise Name'].str.contains('Row|Curl|Bicep|Shrug', case=False, na=False)].copy()
                
                if len(df_filtered) < 50:
                    print(f"   ⚠  Not enough data for {compound} ({len(df_filtered)} rows). Skipping.")
                    continue
                
                # Prepare data
                df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
                df_filtered['date'] = pd.to_datetime(df_filtered['Date'])
                df_filtered['weight'] = pd.to_numeric(df_filtered['Weight'], errors='coerce')
                df_filtered['reps'] = pd.to_numeric(df_filtered['Reps'], errors='coerce')
                df_filtered['set_order'] = pd.to_numeric(df_filtered['Set Order'], errors='coerce')
                
                # Filter to top sets (set_order == 1)
                df_filtered = df_filtered[df_filtered['set_order'] == 1].dropna(subset=['weight', 'reps'])
                
                if len(df_filtered) < 20:
                    print(f"   ⚠  Not enough top sets for {compound}. Skipping.")
                    continue
                
                # Compute load_delta
                df_filtered['load_delta'] = df_filtered['weight'].diff().shift(-1)
                df_filtered = df_filtered.dropna(subset=['load_delta'])
                
                if len(df_filtered) < 10:
                    print(f"   ⚠  Not enough load_delta values for {compound}. Skipping.")
                    continue
                
                # Minimal feature set for robust prediction from user history
                df_filtered['load_delta'] = df_filtered['weight'].diff().shift(-1)
                df_filtered = df_filtered.dropna(subset=['load_delta'])
                if len(df_filtered) < 10:
                    print(f"   ⚠  Not enough training samples for {compound}. Skipping.")
                    continue

                # Ensure RPE column exists
                if 'rpe' not in df_filtered.columns:
                    df_filtered['rpe'] = 8.0
                X = df_filtered[['weight', 'reps', 'rpe']].copy()
                y = df_filtered['load_delta'].astype(float).values
                
                # Build a simple, stable pipeline
                from sklearn.pipeline import Pipeline as SKPipeline
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import r2_score
                
                # Use poor hyperparameters for bench_press to demonstrate rule-based fallback
                if compound == "bench_press":
                    # Intentionally weak model
                    n_est, depth = 10, 1
                else:
                    # Well-trained models
                    n_est, depth = 100, 4
                
                simple_pipe = SKPipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('estimator', RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42))
                ])

                simple_pipe.fit(X, y)
                
                # Compute accuracy (R² score on training data)
                y_pred = simple_pipe.predict(X)
                accuracy = r2_score(y, y_pred)
                accuracy_pct = max(0, int(accuracy * 100))  # Clamp to 0-100%

                # Save model in repo-level models/compounds for app to load
                from joblib import dump
                model_path = models_dir / f"{compound}_model.pkl"
                payload = {"pipeline": simple_pipe, "accuracy": accuracy_pct}
                dump(payload, model_path)
                print(f"   ✓ {compound} model trained on {len(X)} samples, accuracy={accuracy_pct}%, saved to {model_path}")
                    
            except Exception as e:
                print(f"   ⚠  Error training {compound} model: {e}")
                # Continue with other compounds even if one fails
        
        print("\n3. Model training complete!")
        
    except Exception as e:
        print(f"   ✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ DEMO USER SETUP COMPLETE")
    print("=" * 70)
    print("\nDemo User Credentials:")
    print("  Username: user1")
    print("  Password: demo123")
    print("\nModels trained on: strong_4krows_baseline_data.csv (3991 rows)")
    print("Compounds: squat, bench_press, lat_pulldown, seated_row")
    print("\nYou can now login and see recommendations powered by ML models!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = create_demo_user()
    sys.exit(0 if success else 1)
