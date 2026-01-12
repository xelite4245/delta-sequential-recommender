"""Generate recommendations (rule-based or ML) and manage model caching"""
import threading
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from src.rule_based import rule_based_progression
from src.model_quality import is_model_enabled, update_model_quality

# Global model cache (singleton)
class ModelCache:
    """Thread-safe global model cache"""
    _models = {}
    _metadata = {}  # Store accuracy and other metadata
    _lock = threading.Lock()
    
    @classmethod
    def get_model(cls, compound):
        """Load model once, reuse forever in this session"""
        if compound not in cls._models:
            with cls._lock:
                if compound not in cls._models:  # Double-check pattern
                    repo_root = Path(__file__).parent.parent
                    path = repo_root / "models" / "compounds" / f"{compound}_model.pkl"
                    
                    if not path.exists():
                        return None
                    
                    try:
                        model_obj = joblib.load(path)
                        
                        # Extract pipeline and metadata if saved as dict
                        if isinstance(model_obj, dict):
                            cls._models[compound] = model_obj.get('pipeline')
                            cls._metadata[compound] = {
                                'accuracy': model_obj.get('accuracy', 0)
                            }
                        else:
                            cls._models[compound] = model_obj
                            cls._metadata[compound] = {'accuracy': 0}
                    except Exception as e:
                        print(f"Error loading model for {compound}: {e}")
                        return None
        
        return cls._models[compound]
    
    @classmethod
    def get_accuracy(cls, compound):
        """Get model accuracy percentage (0-100)"""
        cls.get_model(compound)  # Ensure model is loaded
        return cls._metadata.get(compound, {}).get('accuracy', 0)
    
    @classmethod
    def clear(cls):
        """Clear cache (only on app exit or model retraining)"""
        cls._models.clear()
        cls._metadata.clear()

def get_recommendation(
    user_id: int,
    user_data_path: str,
    compound: str,
    last_weight: float,
    last_reps: int,
    last_rpe: float,
    session_count: int
) -> Tuple[float, str, str]:
    """
    Generate recommendation (next weight to lift)
    
    Returns: (recommended_weight, source, reason)
    source: 'rule_based' | 'model' | 'insufficient_data'
    """
    
    # Case 1: No data
    if session_count == 0:
        return None, 'insufficient_data', 'No training history yet'
    
    # Case 2: Check if model is explicitly enabled (can bypass 15-session limit)
    model_enabled = is_model_enabled(user_id, compound)
    
    # Case 3: If model NOT enabled and <15 sessions → Always rule-based
    if session_count <= 15 and not model_enabled:
        sugg = rule_based_progression(
            last_weight=last_weight,
            last_reps=last_reps,
            last_rpe=last_rpe
        )
        return sugg.suggested_weight, 'rule_based', sugg.reason
    
    # Case 4: Model is enabled OR 15+ sessions → Try to use ML
    # First, refresh calibration
    try:
        from src.personalized_prediction import maybe_calibrate_affine, CalibrationConfig
        from src.utils.user_personalization import PersonalizationRegistry
        from src.models.base_model import BaseModel
        
        # Load the model
        model_obj = ModelCache.get_model(compound)
        if model_obj:
            # Load history
            user_data_path_obj = Path(user_data_path)
            csv_path = user_data_path_obj / f"{user_data_path_obj.name}_{compound}_history.csv"
            
            if csv_path.exists():
                history = pd.read_csv(csv_path)
                if len(history) > 0:
                    # Create BaseModel wrapper for calibration
                    class ModelWrapper(BaseModel):
                        def __init__(self, pipe):
                            self.pipe = pipe
                        
                        def predict(self, X):
                            return self.pipe.predict(X)
                    
                    wrapped = ModelWrapper(model_obj)
                    registry = PersonalizationRegistry()
                    
                    # Try to refit calibration
                    maybe_calibrate_affine(
                        model=wrapped,
                        registry=registry,
                        user_id=f"User{user_id}",
                        compound=compound,
                        history=history,
                        config=CalibrationConfig()
                    )
    except Exception as e:
        pass  # Silent fail, will use existing calibration
    
    # Update model quality metrics
    try:
        update_model_quality(user_id, compound)
    except Exception as e:
        print(f"Warning: Could not update model quality: {e}")
    
    # Try to use ML model (since we passed the checks above)
    if model_enabled:
        # Try to use ML model
        try:
            pipe = ModelCache.get_model(compound)
            
            if pipe is None:
                # Model not found, fall back to rule-based
                sugg = rule_based_progression(
                    last_weight=last_weight,
                    last_reps=last_reps,
                    last_rpe=last_rpe
                )
                return sugg.suggested_weight, 'rule_based', f"Model not found - {sugg.reason}"
            
            # Get training history
            user_data_path_obj = Path(user_data_path)
            csv_path = user_data_path_obj / f"{user_data_path_obj.name}_{compound}_history.csv"
            
            history = pd.read_csv(csv_path)
            
            if len(history) == 0:
                sugg = rule_based_progression(
                    last_weight=last_weight,
                    last_reps=last_reps,
                    last_rpe=last_rpe
                )
                return sugg.suggested_weight, 'rule_based', f"No history - {sugg.reason}"
            
            # Prepare test row (copy last row and adjust)
            test_row = history.iloc[-1:].copy()
            test_row['reps'] = last_reps
            # Ensure RPE is present; fallback to a neutral value if missing
            try:
                import numpy as np
                if pd.isna(last_rpe):
                    test_row['rpe'] = 8.0
                else:
                    test_row['rpe'] = last_rpe
            except Exception:
                test_row['rpe'] = 8.0
            
            # Drop target column if present in history
            test_row = test_row.drop(columns=['load_delta'], errors='ignore')
            # Get raw prediction
            raw_pred = float(pipe.predict(test_row)[0])
            
            # Apply user calibration via registry
            from src.utils.user_personalization import PersonalizationRegistry
            registry = PersonalizationRegistry()
            up = registry.get_or_create(f"User{user_id}")
            
            # Get adjusted prediction (a*raw + b)
            adjusted_delta = up.adjust_prediction(compound, [raw_pred])[0]
            recommended_weight = last_weight + adjusted_delta
            
            return recommended_weight, 'model', 'ML model (calibrated)'
        
        except Exception as e:
            # Fall back to rule-based
            sugg = rule_based_progression(
                last_weight=last_weight,
                last_reps=last_reps,
                last_rpe=last_rpe
            )
            return sugg.suggested_weight, 'rule_based', f"Model failed - {sugg.reason}"
    
    # Case 4: 15+ sessions but model not enabled yet
    sugg = rule_based_progression(
        last_weight=last_weight,
        last_reps=last_reps,
        last_rpe=last_rpe
    )
    reason = f"Rule-based ({session_count} sessions; model calibrating...)"
    return sugg.suggested_weight, 'rule_based', reason
