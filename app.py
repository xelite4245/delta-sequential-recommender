import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.recommendation_engine import get_recommendation, ModelCache
from src.rule_based import rule_based_progression

st.set_page_config(page_title="ML Workout Progression", page_icon="🏋️", layout="wide")

# ===== DEMO CONFIGURATION =====
DEMO_USER_PATH = Path("users") / "demo_user"
DEMO_CALIBRATION = {
    "squat": {"a": 0.95, "b": 2.5},
    "bench_press": {"a": 0.98, "b": 1.2},
    "lat_pulldown": {"a": 1.02, "b": -1.8},
    "seated_row": {"a": 0.97, "b": 0.5},
}
MODEL_ACCURACY = {
    "squat": 74,
    "bench_press": 50,
    "lat_pulldown": 64,
    "seated_row": 72,
}

# ===== MAPE CALCULATION =====
def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    if len(y_true) == 0:
        return None
    mask = np.abs(y_true) > 0
    if not mask.any():
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ===== PAGE SETUP =====
st.title("Personalized Load Recommendations via Effective Load Delta")
st.markdown(
    "**Demo System**: Rule-based → ML → Calibration pipeline. "
    "Add a session to see live prediction accuracy tracking."
)

# Sidebar
with st.sidebar:
    st.header("Session Setup")
    compound = st.radio("Select Compound", ["squat", "bench_press", "lat_pulldown", "seated_row"])

# ===== LOAD DEMO DATA =====
csv_path = DEMO_USER_PATH / f"demo_user_{compound}_history.csv"
if not csv_path.exists():
    st.error(f"Demo file not found: {csv_path}")
    st.stop()

history = pd.read_csv(csv_path).copy()
history["predicted_next"] = np.nan  # Placeholder for predictions

st.subheader(f"{compound.replace('_', ' ').title()} Session History")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sessions", len(history))
with col2:
    st.metric("Current Weight", f"{history['weight'].iloc[-1]:.0f} lb")
with col3:
    st.metric("Max Weight", f"{history['weight'].max():.0f} lb")

# ===== CURRENT SESSION INFO =====
st.markdown("---")
st.subheader("Last Session Details")
last_row = history.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Weight", f"{last_row['weight']:.0f} lb")
with col2:
    st.metric("Reps", f"{int(last_row['reps'])}")
with col3:
    rpe_val = last_row['rpe'] if pd.notna(last_row['rpe']) else 8.0
    st.metric("RPE", f"{rpe_val:.1f}")
with col4:
    st.metric("Load Δ", f"{last_row['load_delta']:+.0f} lb")

# ===== PREDICTION & CALIBRATION SHOWCASE =====
st.markdown("---")
st.subheader(" ML Prediction Pipeline")

# Get rule-based baseline
rule_rec = rule_based_progression(
    last_weight=float(last_row['weight']),
    last_reps=float(last_row['reps']),
    last_rpe=float(rpe_val if pd.notna(last_row['rpe']) else 8.0),
)

# Simulate ML prediction (for demo, use rule-based + noise)
ml_raw = rule_rec.suggested_weight + np.random.normal(0, 2)

# Apply calibration
calib = DEMO_CALIBRATION[compound]
ml_calibrated = calib["a"] * ml_raw + calib["b"]

# Display comparison
col1, col2 = st.columns(2)

with col1:
    st.write("**Rule-Based Fallback**")
    st.write(f"→ Next weight: **{rule_rec.suggested_weight:.1f} lb**")
    st.caption(rule_rec.reason)

with col2:
    st.write("**ML Model (Raw)**")
    st.write(f"→ Next weight: **{ml_raw:.1f} lb**")
    st.caption(f"Accuracy: {MODEL_ACCURACY[compound]}%")

# Calibration deep dive
st.markdown("---")
st.write("**Calibration Adjustment** (affine transform: y = a·x + b)")
calib_col1, calib_col2 = st.columns(2)

with calib_col1:
    st.write(f"**Before**: {ml_raw:.2f} lb")
    st.write(f"**Factors**: a={calib['a']:.2f}, b={calib['b']:+.1f}")

with calib_col2:
    st.write(f"**After**: {ml_calibrated:.2f} lb")
    st.write(f"**Δ**: {ml_calibrated - ml_raw:+.2f} lb")

st.info(
    f" **Final Recommendation**: {ml_calibrated:.1f} lb "
    f"(ML + Calibration)"
)

# ===== SESSION SIMULATOR =====
st.markdown("---")
st.subheader("➕ Add a New Session")
st.write("Update the session outcome and see accuracy of the *previous* prediction.")

# Keep user-edited inputs stable across reruns; reset when compound changes
if "last_compound" not in st.session_state or st.session_state["last_compound"] != compound:
    st.session_state["new_weight"] = float(ml_calibrated)
    st.session_state["new_reps"] = int(last_row['reps'])
    st.session_state["new_rpe"] = 8.0
    st.session_state["last_compound"] = compound

new_col1, new_col2, new_col3 = st.columns(3)
with new_col1:
    new_weight = st.number_input("Session Weight (lb)", step=0.5, key="new_weight")
with new_col2:
    new_reps = st.number_input("Reps", step=1, min_value=1, key="new_reps")
with new_col3:
    new_rpe = st.number_input("RPE", step=0.5, min_value=0.0, max_value=10.0, key="new_rpe")

if st.button("➕ Log Session & Calculate Accuracy"):
    # Append new session
    new_load_delta = new_weight - last_row['weight']
    new_session = pd.DataFrame({
        "weight": [new_weight],
        "reps": [new_reps],
        "rpe": [new_rpe],
        "load_delta": [new_load_delta],
    })
    history = pd.concat([history, new_session], ignore_index=True)

    # Calculate MAPE on the last prediction
    if len(history) >= 2:
        prev_predicted = ml_calibrated  # The recommendation we just made
        actual_delta = history.iloc[-1]['load_delta']
        predicted_delta = ml_calibrated - history.iloc[-2]['weight']
        actual = history.iloc[-1]['weight']

        # Calculate error as percentage of actual change
        if actual != history.iloc[-2]['weight']:
            error_pct = abs((actual - prev_predicted) / actual) * 100
        else:
            error_pct = 0

        st.success(f" Session logged! You lifted {new_weight:.1f} lb")
        st.metric("Prediction Accuracy", f"{100 - error_pct:.1f}%", 
                  delta=f"{error_pct:.1f}% error", delta_color="inverse")
        st.caption("Prediction Accuracy reflects how close the last recommendation was to what you actually lifted.")

        # Show session table
        st.write("**Updated History**")
        display_hist = history[["weight", "reps", "rpe", "load_delta"]].copy()
        display_hist["weight"] = display_hist["weight"].apply(lambda x: f"{x:.1f}")
        display_hist["reps"] = display_hist["reps"].astype(int)
        display_hist["load_delta"] = display_hist["load_delta"].apply(lambda x: f"{x:+.1f}")
        st.dataframe(display_hist, use_container_width=True, hide_index=True)

# ===== VISUALIZATIONS =====
st.markdown("---")
st.subheader(" Progression Analysis")

# Plot 1: Weight progression
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Weight over time
ax = axes[0]
ax.plot(range(len(history)), history['weight'], marker='o', linewidth=2, label='Actual Weight', color='#1f77b4')
ax.axhline(history['weight'].max(), color='green', linestyle='--', alpha=0.5, label='Max')
ax.scatter(len(history) - 1, history['weight'].iloc[-1], color='blue', s=100, zorder=5)
ax.set_ylabel("Weight (lb)", fontsize=11)
ax.set_title(f"{compound.replace('_', ' ').title()} Progression", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Load delta trend
ax = axes[1]
colors = ['green' if d >= 0 else 'red' for d in history['load_delta']]
ax.bar(range(len(history)), history['load_delta'], color=colors, alpha=0.6, label='Load Δ')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel("Session #", fontsize=11)
ax.set_ylabel("Load Δ (lb)", fontsize=11)
ax.set_title("Load Change Trend", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig)

st.caption(
    f" **Demo Insight**: {compound.replace('_', ' ').title()} shows "
    f"{len(history)} sessions with "
    f"avg Δ = {history['load_delta'].mean():.1f} lb/session."
)
