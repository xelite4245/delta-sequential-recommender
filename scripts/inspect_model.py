import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/compounds/squat_model.pkl')

# Extract the pipeline
if isinstance(model, dict):
    pipe = model.get('pipeline')
else:
    pipe = model

# Show pipeline steps
print("Pipeline steps:")
for name, step in pipe.named_steps.items():
    print(f"  {name}: {step.__class__.__name__}")

# Get feature engineering transformer
feat_eng = pipe.named_steps['feature_engineering']
print(f"\nFeature Engineering config:")
print(f"  n_rolling: {feat_eng.n_rolling}")
print(f"  n_trend: {feat_eng.n_trend}")

# Get the final estimator (RandomForest)
rf = pipe.named_steps['estimator']
print(f"\nRandomForest:")
print(f"  n_estimators: {rf.n_estimators}")
print(f"  Features used: {rf.n_features_in_} (after preprocessing)")

# Try to see feature names if available
try:
    col_transformer = pipe.named_steps['preprocessing']
    print(f"\nColumn Transformer:")
    print(f"  Transformers: {col_transformer.transformers_}")
except:
    pass
