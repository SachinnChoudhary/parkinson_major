"""
Inference service for PD-Aware.

Loads a trained model and provides predictions for new input data.
Validates feature compatibility and records inference metadata.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import MODEL_DIR, DEFAULT_MODEL_VERSION
from ml.train import load_model_artifacts

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Manages model loading and prediction.

    Ensures feature compatibility, applies the correct preprocessing,
    and returns structured prediction results.
    """

    def __init__(self, version: str = None):
        self.version = version or DEFAULT_MODEL_VERSION
        self.model = None
        self.scaler = None
        self.metadata = None
        self._loaded = False

    def load(self) -> bool:
        """Load model artifacts. Returns True if successful."""
        try:
            self.model, self.scaler, self.metadata = load_model_artifacts(self.version)
            self._loaded = True
            logger.info(f"InferenceService loaded model version {self.version}")
            return True
        except FileNotFoundError:
            logger.error(f"Model artifacts not found for version {self.version}")
            self._loaded = False
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def expected_features(self) -> List[str]:
        """Return the list of features the model expects."""
        if self.metadata:
            return self.metadata.get("features", [])
        return []

    def validate_features(self, feature_dict: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that input features match the model's expected schema.

        Returns (is_valid, list_of_issues).
        """
        if not self._loaded:
            return False, ["Model not loaded"]

        expected = set(self.expected_features)
        provided = set(feature_dict.keys())

        issues = []

        missing = expected - provided
        if missing:
            issues.append(f"Missing features: {sorted(missing)}")

        extra = provided - expected
        if extra:
            issues.append(f"Unexpected features (will be ignored): {sorted(extra)}")

        # Check for non-numeric values
        for key, val in feature_dict.items():
            if key in expected and not isinstance(val, (int, float, np.floating, np.integer)):
                issues.append(f"Feature '{key}' has non-numeric value: {val}")

        # Check for NaN
        for key, val in feature_dict.items():
            if key in expected and isinstance(val, float) and np.isnan(val):
                issues.append(f"Feature '{key}' is NaN")

        is_valid = len(missing) == 0 and not any("non-numeric" in i for i in issues)
        return is_valid, issues

    def predict(self, feature_dict: Dict[str, float]) -> Dict:
        """
        Generate a prediction for the given features.

        Returns a structured result with prediction, score, metadata,
        and timing information.
        """
        if not self._loaded:
            return {
                "success": False,
                "error": "Model not loaded. Please train or load a model first.",
            }

        # Validate features
        is_valid, issues = self.validate_features(feature_dict)
        if not is_valid:
            return {
                "success": False,
                "error": "Feature validation failed",
                "issues": issues,
            }

        start_time = time.time()

        try:
            # Extract features in the correct order
            feature_values = np.array([
                [feature_dict.get(f, 0.0) for f in self.expected_features]
            ])

            # Apply preprocessing (same scaler used during training)
            feature_scaled = self.scaler.transform(feature_values)

            # Predict
            prediction_label = int(self.model.predict(feature_scaled)[0])

            # Get probability if available
            prediction_score = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(feature_scaled)[0]
                prediction_score = float(proba[1])  # Probability of class 1 (PD)
            elif hasattr(self.model, "decision_function"):
                prediction_score = float(self.model.decision_function(feature_scaled)[0])

            inference_time_ms = (time.time() - start_time) * 1000

            # Map to human-readable category
            target_encoding = self.metadata.get("target_encoding", {})
            prediction_category = target_encoding.get(
                str(prediction_label),
                "parkinson" if prediction_label == 1 else "healthy"
            )

            result = {
                "success": True,
                "prediction_label": prediction_label,
                "prediction_score": round(prediction_score, 4) if prediction_score is not None else None,
                "prediction_category": prediction_category,
                "model_version": self.version,
                "model_type": self.metadata.get("model_type", "unknown"),
                "feature_schema_version": self.metadata.get("feature_schema_version", "1.0"),
                "preprocessing_version": self.metadata.get("preprocessing_version", "1.0"),
                "inference_time_ms": round(inference_time_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_features": {f: round(v, 6) for f, v in feature_dict.items()
                                   if f in self.expected_features},
                "feature_names": self.expected_features,
                "issues": issues if issues else [],
                "limitations": self.metadata.get("known_limitations", []),
            }

            logger.info(
                f"Prediction: label={prediction_label}, score={prediction_score}, "
                f"time={inference_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
            }

    def get_model_info(self) -> Dict:
        """Return model metadata for display."""
        if not self._loaded:
            return {"loaded": False, "error": "No model loaded"}

        return {
            "loaded": True,
            "version": self.version,
            "model_type": self.metadata.get("model_type", "unknown"),
            "model_name": self.metadata.get("model_name", "unknown"),
            "dataset": self.metadata.get("dataset", {}),
            "metrics": self.metadata.get("metrics", {}),
            "training_date": self.metadata.get("training_date", "unknown"),
            "intended_use": self.metadata.get("intended_use", ""),
            "known_limitations": self.metadata.get("known_limitations", []),
            "status": self.metadata.get("status", "unknown"),
            "n_features": len(self.expected_features),
            "features": self.expected_features,
        }


# ─── Singleton-like access ───────────────────────────────────────────────────

_service: Optional[InferenceService] = None


def get_inference_service(version: str = None) -> InferenceService:
    """Get or create the inference service singleton."""
    global _service
    if _service is None or (version and _service.version != version):
        _service = InferenceService(version)
        _service.load()
    return _service
