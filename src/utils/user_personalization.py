"""
Per-user personalization for compound models.

Instead of retraining models per user, apply lightweight adjustments:
- User scaling factors (e.g., user might progress slower on bench)
- User baseline offsets (e.g., consistent over/under prediction)
- User trend modifiers (e.g., accelerating vs. stalling progression)

These are stored per user and refined over time as more data arrives.
"""

from pathlib import Path
from typing import Dict, Optional
import json
import pandas as pd
import numpy as np


class UserPersonalization:
    """
    Per-user adjustments to compound model predictions.
    
    Attributes:
        user_id: Unique user identifier
        scaling_factors: Dict[compound -> factor] (1.0 = default)
        baseline_offsets: Dict[compound -> offset] (0.0 = default)
        trend_modifiers: Dict[compound -> modifier] (1.0 = default)
    """
    
    def __init__(
        self,
        user_id: str,
        scaling_factors: Optional[Dict[str, float]] = None,
        baseline_offsets: Optional[Dict[str, float]] = None,
        trend_modifiers: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize user personalization.
        
        Args:
            user_id: User identifier
            scaling_factors: Dict[compound -> multiplier] for load_delta
            baseline_offsets: Dict[compound -> additive offset] for load_delta
            trend_modifiers: Dict[compound -> multiplier] for trend strength
        """
        self.user_id = user_id
        self.scaling_factors = scaling_factors or self._default_dict(1.0)
        self.baseline_offsets = baseline_offsets or self._default_dict(0.0)
        self.trend_modifiers = trend_modifiers or self._default_dict(1.0)
    
    @staticmethod
    def _default_dict(value: float) -> Dict[str, float]:
        """Create default dict for all compounds."""
        return {
            "squat": value,
            "bench_press": value,
            "lat_pulldown": value,
            "seated_row": value,
        }
    
    def adjust_prediction(
        self,
        compound: str,
        raw_prediction: float
    ) -> float:
        """
        Apply user personalization to a raw model prediction.
        
        Formula:
            adjusted = (raw_prediction * scaling_factor) + baseline_offset
        
        Args:
            compound: Parent compound name
            raw_prediction: Raw model prediction (load_delta)
            
        Returns:
            Adjusted prediction
        """
        scaling = self.scaling_factors.get(compound, 1.0)
        offset = self.baseline_offsets.get(compound, 0.0)
        
        adjusted = (raw_prediction * scaling) + offset
        return adjusted
    
    def learn_from_residuals(
        self,
        compound: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        update_rate: float = 0.05
    ) -> None:
        """
        Learn adjustments from prediction residuals.
        
        Incremental learning: shift baseline_offset and scaling_factor
        based on observed over/under prediction.
        
        Args:
            compound: Parent compound name
            y_true: True load_delta values
            y_pred: Model predictions
            update_rate: Learning rate (0.0-1.0)
        """
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        
        # Shift baseline offset
        old_offset = self.baseline_offsets[compound]
        self.baseline_offsets[compound] = (
            old_offset + (mean_residual * update_rate)
        )
    
    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "user_id": self.user_id,
            "scaling_factors": self.scaling_factors,
            "baseline_offsets": self.baseline_offsets,
            "trend_modifiers": self.trend_modifiers,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserPersonalization":
        """Deserialize from dict."""
        return cls(
            user_id=data["user_id"],
            scaling_factors=data.get("scaling_factors"),
            baseline_offsets=data.get("baseline_offsets"),
            trend_modifiers=data.get("trend_modifiers"),
        )
    
    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "UserPersonalization":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class PersonalizationRegistry:
    """
    Manager for per-user personalization across all users.
    
    Handles loading, saving, and per-user adjustments.
    """
    
    def __init__(self, base_dir: Path = Path("users")):
        """
        Initialize registry.
        
        Args:
            base_dir: Root directory for user data
        """
        self.base_dir = Path(base_dir)
        self.personalizations: Dict[str, UserPersonalization] = {}
    
    def get_or_create(self, user_id: str) -> UserPersonalization:
        """Get user personalization, creating if needed."""
        if user_id not in self.personalizations:
            # Try to load from disk
            path = self._get_path(user_id)
            if path.exists():
                self.personalizations[user_id] = UserPersonalization.load(path)
            else:
                self.personalizations[user_id] = UserPersonalization(user_id)
        
        return self.personalizations[user_id]
    
    def save(self, user_id: str) -> None:
        """Save user personalization to disk."""
        if user_id in self.personalizations:
            path = self._get_path(user_id)
            self.personalizations[user_id].save(path)
    
    def save_all(self) -> None:
        """Save all personalizations."""
        for user_id in self.personalizations:
            self.save(user_id)
    
    def _get_path(self, user_id: str) -> Path:
        """Get JSON path for user personalization."""
        return self.base_dir / user_id / "personalization.json"
