# Personalized Load Recommendations via Effective Load Delta

Delta is a strength progression system that predicts next-session training loads under sparse, noisy data, using gated machine-learning models with deterministic fallbacks to ensure safe and reliable recommendations.

## Problem Statement

Strength progression data is scattered, noisy, and highly individual. Most users log fewer than 50 sessions per compound, making standard ML models unreliable without aggressive regularization. Rule-based progressions lack personalization and adapt slowly to individual behavior. This project explores how to combine ML and deterministic logic under real-world data scarcity.

## Approach

Built personalization on top of scattered user data:
- **Train separate models per compound lift** (squat, bench, row, etc.) on ~11k cross-user historical sessions, capturing lift-specific progression patterns.
- **Personalize via calibration**: adjust raw predictions to each user using their recent sessions (`adjusted = a × prediction + b`). The model learns individual pace within 8–10 logged sessions per compound.
- **Validate aggressively**: use cross-validation before deployment. Treat negative or volatile CV scores as a hard stop—route predictions to deterministic logic instead of forcing ML output.
- **Serve predictions** only when we have 15+ sessions per compound (otherwise pure rule-based). Update calibration incrementally as new data arrives.

## Modeling & Validation

Tree-based models (XGBoost, Random Forest) were evaluated on 30–50 session validation splits. XGBoost achieved 93% training R² but exhibited unstable cross-validation (−135% CV R²), demonstrating overfitting. Random Forest showed stronger CV stability (−38% mean R²) and was selected for deployment. Negative or volatile CV scores are treated as a hard stop.

| Model | Training R² | CV Mean R² | Decision |
|-------|-------------|-----------|----------|
| XGBoost | 93% | −135% | Gated (overfitting) |
| Random Forest | 76% | −38% | Conditional |
| Linear | 13% | −53% | Rejected |

## Safety & Fallback Logic

- Disable ML predictions below 15 historical sessions per compound.
- Reject models with negative or unstable cross-validation scores.
- Default to conservative heuristics: cap increases at 5–10 lbs per session, require 3+ reps before load increase, detect deloads (15%+ drop).
- Prevent unsafe load jumps by clamping deltas relative to recent session variance.
- Enforce model gating: require ≥8 calibration samples and ≥10 CV runs before enabling ML for a user-compound pair.

## Demo / Usage

- **Demonstrates ML gating vs deterministic fallback** under sparse data: toggle between model-based and rule-based predictions to see when each is used.
- **Visualizes prediction error and calibration behavior** over sequential sessions: track MAPE, see how per-user adjustments evolve.
- **Streamlit app** (`app.py`): Interactive demo with sidebar compound selector and session-by-session error tracking.
- **Pre-loaded demo user** with squat/bench/row history to explore immediately.
- **Simulate new sessions**: Add weight/reps and observe how previous predictions performed.
- **Show calibration coefficients** (before/after adjustments for each lift).

Run locally:
```bash
streamlit run app.py
```

## Project Structure

- **delta-sequential-recommender/**
  - **app.py**: Streamlit demo (main entry point)
  - **src/**: recommendation_engine (ML routing, calibration), rule_based (fallback heuristics), personalized_prediction (per-user calibration), and supporting modules
  - **models/compounds/**: Trained Random Forest pipelines per lift
  - **data/baseline/**: ~11k historical training sessions
  - **notebooks/**: Exploratory analysis and walkthroughs
  - **users/**: Demo and test data
  - **tests/**: Workflow and model validation

## Limitations

- **Small per-user datasets** limit ML generalization; cross-validation often yields negative R² on validation splits <30 samples.
- **Fixed compound set** (squat, bench press, lat pulldown, seated row); requires manual feature engineering and retraining for new lifts.
- **Demo uses historical baseline data**, not live user-generated sessions; real-world performance depends on data quality and consistency.

## Future Work

- Incorporate Bayesian updating to improve early-session personalization (first 5–10 sessions).
- Detect and model periodization cycles (deload detection, rep-range shifts).
- Expand to isolation exercises with shared feature representations.

---

**For model evaluation details**, see [model_scores.txt](https://github.com/azizuddinuzair/delta-sequential-recommender/blob/main/tests/model_scores.txt).
