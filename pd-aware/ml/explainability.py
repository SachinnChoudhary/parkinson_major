"""
SHAP-based model explainability for PD-Aware.

Generates local (per-prediction) and global feature importance explanations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import shap

from config.settings import VOICE_FEATURES

logger = logging.getLogger(__name__)

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
    "MDVP:Jitter(%)": "Frequency variation (jitter percentage)",
    "MDVP:Jitter(Abs)": "Frequency variation (absolute jitter)",
    "MDVP:RAP": "Relative amplitude perturbation",
    "MDVP:PPQ": "Five-point period perturbation quotient",
    "Jitter:DDP": "Average absolute difference of differences between cycles",
    "MDVP:Shimmer": "Amplitude variation (shimmer)",
    "MDVP:Shimmer(dB)": "Amplitude variation in decibels",
    "Shimmer:APQ3": "Three-point amplitude perturbation quotient",
    "Shimmer:APQ5": "Five-point amplitude perturbation quotient",
    "MDVP:APQ": "Eleven-point amplitude perturbation quotient",
    "Shimmer:DDA": "Average absolute differences between amplitudes",
    "NHR": "Noise-to-harmonics ratio",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy",
    "DFA": "Detrended fluctuation analysis",
    "spread1": "Nonlinear measure of fundamental frequency variation",
    "spread2": "Nonlinear measure of fundamental frequency variation",
    "D2": "Correlation dimension",
    "PPE": "Pitch period entropy",
}


def explain_prediction(
    model,
    scaler,
    feature_values: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
) -> Dict:
    """
    Generate SHAP-based explanation for a single prediction.

    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_values: Raw feature values (1, n_features) — UNSCALED
        feature_names: List of feature names
        background_data: Background dataset for SHAP (scaled). If None, uses
                        a small synthetic background.

    Returns:
        Dictionary with SHAP values, feature contributions, and descriptions.
    """
    try:
        # Scale the input
        feature_scaled = scaler.transform(feature_values)

        # Create background data if not provided
        if background_data is None:
            # Use a small synthetic background (mean ± 1 std for each feature)
            n_bg = 50
            rng = np.random.RandomState(42)
            background_data = rng.randn(n_bg, len(feature_names))

        # Choose appropriate SHAP explainer
        model_type = type(model).__name__.lower()

        if "tree" in model_type or "forest" in model_type or "xgb" in model_type or "gradient" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for linear/SVM models
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                background_data[:min(50, len(background_data))],
            )

        # Compute SHAP values
        shap_values = explainer.shap_values(feature_scaled)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: use class 1 (PD) SHAP values
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                sv = shap_values[0, :, 1]  # (samples, features, classes)
            elif shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = shap_values
        else:
            # shap.Explanation object
            if hasattr(shap_values, 'values'):
                sv = shap_values.values[0]
                if sv.ndim > 1:
                    sv = sv[:, 1]
            else:
                sv = np.zeros(len(feature_names))
                logger.warning("Could not extract SHAP values")

        # Build explanation
        contributions = []
        for i, fname in enumerate(feature_names):
            shap_val = float(sv[i])
            raw_val = float(feature_values[0, i])
            contributions.append({
                "feature": fname,
                "description": FEATURE_DESCRIPTIONS.get(fname, fname),
                "raw_value": round(raw_val, 4),
                "shap_value": round(shap_val, 4),
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
                "magnitude": abs(round(shap_val, 4)),
            })

        # Sort by magnitude (most important first)
        contributions.sort(key=lambda x: x["magnitude"], reverse=True)

        return {
            "success": True,
            "shap_values": [round(float(v), 4) for v in sv],
            "feature_names": feature_names,
            "contributions": contributions,
            "top_contributors": contributions[:5],
            "explanation_method": "SHAP (SHapley Additive exPlanations)",
            "note": (
                "SHAP values show how each feature pushed the model's output "
                "compared to the average prediction. Positive values indicate "
                "the feature increased the model's Parkinson's-associated score; "
                "negative values indicate it decreased the score. "
                "This does not prove medical causation."
            ),
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "note": "Feature importance explanation could not be generated for this prediction.",
        }


def get_global_feature_importance(model, feature_names: List[str]) -> Dict:
    """
    Get global feature importance from the model.

    Uses built-in feature_importances_ for tree models,
    or coefficient magnitudes for linear models.
    """
    importances = []

    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return {"success": False, "error": "Model does not support feature importance"}

    for i, fname in enumerate(feature_names):
        importances.append({
            "feature": fname,
            "description": FEATURE_DESCRIPTIONS.get(fname, fname),
            "importance": round(float(raw_importances[i]), 4),
        })

    importances.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "success": True,
        "importances": importances,
        "method": "Model built-in feature importance",
        "top_features": importances[:10],
    }
