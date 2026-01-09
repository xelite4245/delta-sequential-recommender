# Model Assumptions & Target User Profile

## Current Design: Intermediate/Advanced Lifters

The Personalized Workout Progression System is optimized for **intermediate to advanced lifters** with established training patterns, periodization, and deload cycles. This document explains why and what happens with beginners.

---

## Key Findings

### Periodization Features (Scale-Invariant)

The model detects training cycles using **percentage-based deload detection**:

```python
is_deload = (prev_weight - current_weight) / prev_weight >= 0.15  # 15% drop
```

This makes the model scale-invariant:
- **Beginner (55 lbs)**: 15% drop = ~8 lbs
- **Intermediate (200 lbs)**: 15% drop = ~30 lbs  
- **Advanced (400 lbs)**: 15% drop = ~60 lbs

### Training Data Profile

The PPL training data is from **heavy lifters (135–425 lbs squats)**:

| Compound | Weight Range | Max Drop | Sample Size (Top Sets) |
|----------|--------------|----------|------------------------|
| Squat    | 40–425 lbs   | -320 lbs | 469                    |
| Bench    | 50–315 lbs   | -310 lbs | 863                    |
| Pulldown | 30–280 lbs   | -250 lbs | 431                    |

---

## Example: User2 (Beginner)

### Profile
- **Weight range**: 45–75 lbs squats
- **Pattern**: PR (65 lbs) → deload (45 lbs) → climb back up
- **Deload magnitude**: 20 lbs = 31% drop (detected ✓)

### Model Output
When at **week 3 of climb, 73% of max**, the model predicts:
- **Raw**: -32 lbs (drop to ~23 lbs)
- **After calibration**: -18 lbs (drop to ~37 lbs)

### Why It Happens
The PPL training data has **very few samples** at "week 3, 70–75% of max" state. The model extrapolates poorly for beginner-scale progressions, leading to conservative (pessimistic) predictions.

### How Calibration Helps
After User2 logs 20–30 more sessions, the fitted affine calibration (a, b) learns:
- Global model tends to underestimate User2's climbing ability
- Calibration applies a corrective gain (`a`) and offset (`b`)
- Future predictions converge toward User2's actual progression

---

## Recommendations

### Use This Model For
- ✅ Intermediate lifters (6+ months training, established patterns)
- ✅ Advanced lifters (1+ year, solid periodization)
- ✅ Any user after 8–12 weeks of logged history (calibration converges)

### Don't Use For
- ❌ Complete beginners (0–3 months) without significant history
- ❌ Users with highly irregular or non-periodized training
- ❌ Exercises without enough training data in the PPL dataset

### Alternative for Beginners
1. **Rule-based fallback** (conservative, safe)
   ```bash
   python -m src.cli predict --bench-fallback ...
   ```
   Uses simple heuristics: drop if too hard, hold near limit, bump if easy.

2. **Accumulate history first**
   - Log 8–12 sessions with `load_delta` labels
   - Then run `refresh-calibration` to fit User2-specific (a, b)
   - Predictions improve significantly after this

3. **Separate beginner model** (future)
   - Train on User2-scale data (45–100 lbs)
   - Shorter deload cycles, higher growth rates

---

## Technical Details

### Periodization Features

All models engineer these cycle-aware features:

| Feature | Formula | Interpretation |
|---------|---------|-----------------|
| `is_deload` | 15% weight drop | Entering recovery phase |
| `cycle_number` | Cumulative deload count | Which training block |
| `weeks_in_cycle` | Sessions since deload | Phase of cycle (1=entry, 8+=plateau) |
| `max_weight_so_far` | Expanding max | Historical PR |
| `percent_of_max` | weight / max_weight | Proximity to recent PR (0–100%) |
| `cycle_weight_trend` | Weight change within cycle | Local slope |

### Feature Scaling

The preprocessor pipeline normalizes all features:
```
Pipeline:
  1. FeatureEngineeringTransformer (add cycles)
  2. ColumnTransformer (impute + scale numeric, onehot categorical)
  3. RandomForestRegressor (predictions)
```

---

## Calibration Convergence

User2's calibration history:

| Run | History Size | Calibration | a (gain) | b (offset) | Confidence |
|-----|--------------|-------------|----------|-----------|------------|
| 1   | 299 rows     | Fitted      | 0.60     | +1.23     | High       |
| (Future) | 320+ rows | Refitted | ? | ? | Higher |

The affine transform `adjusted = a * raw + b` learns:
- `a < 1.0`: Model is too aggressive; dampen predictions
- `b > 0`: Model is pessimistic; add baseline optimism
- As history grows, (a, b) stabilize toward User2's true progression

---

## Deployment Notes

### SQLite Logging

Predictions are logged to `data/user_data.db`:

```sql
SELECT user_id, compound, predicted_raw, predicted_adjusted, source, created_at
FROM predictions
WHERE user_id = 'User2' AND compound = 'squat'
ORDER BY created_at DESC LIMIT 5;
```

Use this to:
- Track model accuracy over time (if you log actuals later)
- Audit which predictions went to which users
- Detect systematic biases per user/compound

### JSON Personalization

User calibration stored in `users/{user_id}/personalization.json`:

```json
{
  "scaling_factors": { "squat": 0.6, ... },
  "baseline_offsets": { "squat": 1.23, ... },
  "calibration_meta": {
    "squat": { "last_calibrated_size": 32, "runs": 1 }
  }
}
```

Loaded once per user and cached in `PersonalizationRegistry` for fast lookups.

---

## Future Improvements

1. **Adaptive deload threshold**: Learn per-user deload % from history
2. **Beginner-specific model**: Separate RF trained on 0–6 month progressions
3. **Strength standards normalization**: Scale features by body weight + experience level
4. **Actuals logging**: Post-hoc accuracy tracking to validate calibration
5. **Hierarchical models**: User-level features (e.g., days/week training) + compound-level features

---

## Summary

The model works well for intermediate/advanced lifters with established periodization. For beginners, **calibration + 3–4 weeks of history** is sufficient to converge. The percentage-based deload detection (15%) ensures the model adapts across all strength levels.
