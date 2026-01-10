# Interactive CLI Application

This is a user-friendly command-line application for the Personalized Workout Progression System.

## Quick Start

```bash
python run_app.py
```

## Features

### 1. **User Authentication**
- Login with existing account
- Create new account (automatically initializes user data)
- Plaintext passwords (MVP security)

### 2. **Session Logging**
- Select which lift to log (squat, bench press, lat pulldown, seated row)
- Enter weight, reps, and RPE
- Specify how the session felt:
  - **Normal**: As expected
  - **Easy**: Could have done more
  - **Hard**: Struggled more than expected
  - **Injury/Pain**: Limited by pain
  - **External Stress**: Sleep issues, life stress, etc.
  - **Other**: Add custom notes

### 3. **Intelligent Recommendations**
The system automatically selects the best prediction method:

- **0 sessions**: No recommendation yet
- **1-15 sessions**: Rule-based progression (consistent, deterministic)
- **15+ sessions with low quality**: Continue rule-based
- **15+ sessions with good quality**: ML model + calibration

### 4. **Automatic Accuracy Tracking**
- When you log a new session, the app automatically checks if previous predictions were correct
- Marks predictions as "pending" → "complete" with accuracy delta
- Uses only "normal" deviation sessions to calculate model quality (filters out injury/stress)

### 5. **Model Quality Detection**
- Calculates MAPE (Mean Absolute Percentage Error) on recent sessions
- Compares ML model vs rule-based performance
- Auto-enables model when it's 15% better than rule-based AND under 10% error
- Safe fallback: Always falls back to rule-based if model fails

### 6. **Calibration**
- Per-user affine transformation (adjusted = a × raw + b)
- Auto-refreshed on every session
- Ensures recommendations match your strength level
- Coefficients clamp to [0.6, 1.4] to prevent wild predictions

## Database Schema

### `data/auth/app_users.db`
- **users**: Username, password, user_data_path
- **model_quality**: Session count, MAPE, model enabled status per user/compound
- **session_audit**: Full audit trail of predictions vs actuals

### User Data
Each user has:
- `users/{username}/{username}_{compound}_history.csv` - Training history
- `users/{username}/personalization.json` - Calibration coefficients (a, b)
- `users/{username}/plots/` - Generated visualizations

## Workflow Example

```
Welcome to Personalized Workout Progression!

1. Login
2. Sign Up (New Account)
3. Exit

Select (1-3): 2
Username: alice
Password: •••••
Confirm:  •••••

✓ Account created! Welcome, alice!

What lift to log today?
1. Squat
2. Bench Press
3. Lat Pulldown
4. Seated Row
5. Exit

Select: 1

📋 Log Session: Squat
Weight: 185
Reps: 5
RPE: 7

How did this set feel?
1. Normal (as expected)
2. Easy (could have done more)
3. Hard (struggled more than expected)
4. Injury/Pain (limited by pain)
5. External Stress (sleep, stress, etc.)
6. Other (please add notes)

Select: 1

✓ Session logged!

📊 Next Session Recommendation:
Compound: Squat
Current: 185 lbs × 5 reps
Recommended: 190 lbs
Change: +5 lbs

Prediction Method: RULE_BASED
Reason: First session - log more to enable predictions

Progress: 1 session logged
Log 14 more sessions to enable ML predictions

[Log another session? (y/n)]
```

## Architecture

### Core Modules

**`src/auth.py`**
- `login(username, password)` → (user_id, user_data_path)
- `register(username, password)` → (user_id, user_data_path)
- `get_user_id(username)` → user_id

**`src/ui.py`**
- `login_screen()` → (username, password, choice)
- `compound_menu()` → compound name
- `log_session_menu(compound)` → (weight, reps, rpe, deviation_reason)
- `show_recommendation(...)` → Display prediction

**`src/session_logger.py`**
- `log_session(...)` → prediction_id (stored as "pending")
- `compute_accuracy_for_pending_predictions(...)` → Mark previous predictions "complete" with accuracy
- `get_session_count(user_path, compound)` → int
- `get_last_session(user_path, compound)` → (weight, reps, rpe)

**`src/model_quality.py`**
- `calculate_mape(actual, predicted)` → float
- `update_model_quality(user_id, compound)` → bool (enabled or not)
- `is_model_enabled(user_id, compound)` → bool
- `get_session_count(user_id, compound)` → int

**`src/recommendation_engine.py`**
- `ModelCache` - Singleton for caching loaded models in memory
- `get_recommendation(...)` → (recommended_weight, source, reason)

## Decision Logic

```python
if session_count == 0:
    # No recommendation
    return None

elif session_count <= 15:
    # Always rule-based for first 15 sessions
    return rule_based_progression(...)

else:
    # 15+ sessions: Check model quality
    update_model_quality(user_id, compound)
    
    if is_model_enabled(user_id, compound):
        # ML is better than rule-based
        return ml_prediction(...)
    else:
        # Still gathering data or ML hasn't converged
        return rule_based_progression(...)
```

## Deviation Reasons

The app captures **why** you deviated from the prediction:

| Reason | Meaning | Impact on Model Quality |
|--------|---------|------------------------|
| normal | As expected | ✓ Used to calculate MAPE |
| easy | Could have done more | ✗ Filtered out (not model's fault) |
| hard | Struggled more | ✗ Filtered out (not model's fault) |
| injury | Limited by pain | ✗ Filtered out (not model's fault) |
| external_stress | Sleep, stress, etc. | ✗ Filtered out (not model's fault) |
| other | Custom notes | ✗ Filtered out (not model's fault) |

**Only "normal" sessions are used to calculate MAPE and determine if the model is good.**

## Model Caching

Models are loaded once and cached in memory for the entire session:

```python
# First prediction: Load from disk (~200ms)
pipe = ModelCache.get_model("squat")

# Subsequent predictions: Use cached copy (~1ms)
pipe = ModelCache.get_model("squat")  # Instant
```

## Accuracy Tracking Workflow

```
Timeline:
  Session 1: Log 185 × 5
    → Predict: 190 lbs (rule-based, pending)
  
  Session 2: Log 188 × 5
    → Compute accuracy: predicted=190, actual=188, delta=-2 lbs
    → Mark prediction 1 as "complete"
    → Predict: 192 lbs (rule-based, pending)
  
  Session 3+: Rinse and repeat
```

**Result:** By session 15-20, you have 10-15 "complete" predictions to evaluate model quality.

## Troubleshooting

### "User data path does not exist"
The user folder was deleted. Try creating a new account or restoring the user directory.

### "Model not found"
Run `python -m src.cli train-compounds` to train models first.

### "Database error"
Delete `data/auth/app_users.db` and run `python data/auth/init_auth_db.py` to reinitialize.

### Model stays "rule-based" after 15+ sessions
The ML model hasn't converged yet. It needs:
- At least 10 completed predictions
- MAPE < rule-based MAPE × 0.85
- MAPE < 10% absolute error

Keep logging sessions and it will auto-enable when criteria are met.

## Future Enhancements

- TKinter GUI (user offered to build this)
- Multi-model approach (separate injury/recovery models)
- RPE trend analysis
- Periodization hints
- Export/visualization commands
