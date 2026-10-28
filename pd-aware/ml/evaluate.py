"""
Model evaluation module for PD-Aware.

Computes classification metrics, generates confusion matrices,
ROC curves, and comparison reports for all trained models.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> Dict:
    """
    Evaluate a trained model on the test set.

    Returns a dictionary of metrics including accuracy, precision, recall,
    F1, ROC-AUC, PR-AUC, sensitivity, specificity, and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    # Core metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),  # Same as recall
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Specificity (TNR)
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
    else:
        metrics["specificity"] = 0.0

    metrics["confusion_matrix"] = cm.tolist()

    # Classification report as dict
    metrics["classification_report_dict"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0,
    )

    # AUC metrics (require probability scores)
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0
            logger.warning(f"{model_name}: Could not compute ROC-AUC")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
        except ValueError:
            metrics["pr_auc"] = 0.0

        # ROC curve data
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }
        except Exception:
            pass

        # PR curve data
        try:
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
            metrics["pr_curve"] = {
                "precision": prec_curve.tolist(),
                "recall": rec_curve.tolist(),
            }
        except Exception:
            pass
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        logger.warning(f"{model_name}: No probability scores available for AUC")

    return metrics


def generate_evaluation_report(results: Dict, feature_names: List[str]) -> Dict:
    """
    Generate a comparative evaluation report for all models.

    Returns structured data suitable for display in the UI.
    """
    comparison = []

    for model_key, result in results.items():
        if "error" in result:
            comparison.append({
                "model_key": model_key,
                "name": model_key,
                "status": "error",
                "error": result["error"],
            })
            continue

        m = result["metrics"]
        comparison.append({
            "model_key": model_key,
            "name": result["name"],
            "status": "trained",
            "accuracy": m.get("accuracy", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "sensitivity": m.get("sensitivity", 0),
            "specificity": m.get("specificity", 0),
            "f1": m.get("f1", 0),
            "roc_auc": m.get("roc_auc", 0),
            "pr_auc": m.get("pr_auc", 0),
            "train_time": result.get("train_time_seconds", 0),
        })

    # Sort by composite score (same as select_best_model)
    trained = [c for c in comparison if c["status"] == "trained"]
    for c in trained:
        c["composite_score"] = round(
            0.4 * c["roc_auc"] + 0.3 * c["f1"] + 0.3 * c["sensitivity"], 4
        )

    trained.sort(key=lambda x: x["composite_score"], reverse=True)

    return {
        "model_comparison": trained,
        "n_models_trained": len(trained),
        "n_models_failed": len(comparison) - len(trained),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "best_model": trained[0] if trained else None,
        "ranking_criteria": "Composite: 0.4×ROC-AUC + 0.3×F1 + 0.3×Sensitivity",
    }
