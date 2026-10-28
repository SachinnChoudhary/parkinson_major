"""
Model training pipeline for PD-Aware.

Trains multiple classifiers on the UCI Parkinson's dataset with subject-aware
splitting and proper preprocessing (scaler fit only on training data).

Persists trained models, scalers, and metadata as versioned artifacts.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config.settings import MODEL_DIR, VOICE_FEATURES, DEFAULT_MODEL_VERSION
from ml.dataset import (
    download_dataset, load_dataset, inspect_dataset,
    subject_aware_split, preprocess, extract_subject_id,
)
from ml.evaluate import evaluate_model, generate_evaluation_report

logger = logging.getLogger(__name__)


# ─── Model Definitions ───────────────────────────────────────────────────────

def get_model_configs() -> Dict:
    """Define candidate models with hyperparameters."""
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
                class_weight="balanced",
            ),
            "description": "Linear model, interpretable, fast",
        },
        "svm": {
            "name": "Support Vector Machine",
            "model": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=42,
                class_weight="balanced",
            ),
            "description": "Non-linear classifier, good for small datasets",
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "description": "Ensemble of decision trees, handles non-linearities",
        },
        "extra_trees": {
            "name": "Extra Trees",
            "model": ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "description": "Extremely randomized trees, reduced variance",
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "model": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42,
            ),
            "description": "Sequential boosting, strong predictive power",
        },
        "xgboost": {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=1.0,
            ),
            "description": "Optimized gradient boosting, often best performance",
        },
    }


# ─── Training Pipeline ───────────────────────────────────────────────────────

def train_single_model(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> Tuple:
    """Train a single model and return it with training metadata."""
    configs = get_model_configs()
    if model_key not in configs:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(configs.keys())}")

    config = configs[model_key]
    model = config["model"]

    logger.info(f"Training {config['name']}...")
    start_time = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    logger.info(f"  Trained in {train_time:.2f}s")

    return model, train_time


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Train all candidate models and evaluate them."""
    configs = get_model_configs()
    results = {}

    for model_key, config in configs.items():
        try:
            model, train_time = train_single_model(
                model_key, X_train, y_train, feature_names,
            )

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test, model_name=config["name"])

            results[model_key] = {
                "model": model,
                "name": config["name"],
                "description": config["description"],
                "metrics": metrics,
                "train_time_seconds": round(train_time, 3),
            }

            logger.info(
                f"  {config['name']}: accuracy={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to train {config['name']}: {e}")
            results[model_key] = {"error": str(e)}

    return results


def select_best_model(results: Dict) -> Tuple[str, Dict]:
    """
    Select the best model based on a composite score.

    We prioritize:
    1. ROC-AUC (most robust for imbalanced data)
    2. F1 score (harmonic mean of precision and recall)
    3. Sensitivity/Recall (critical for screening — minimize false negatives)
    """
    valid_results = {
        k: v for k, v in results.items() if "error" not in v
    }

    if not valid_results:
        raise ValueError("No models trained successfully")

    # Composite score: 0.4*AUC + 0.3*F1 + 0.3*Sensitivity
    scores = {}
    for key, result in valid_results.items():
        m = result["metrics"]
        composite = 0.4 * m["roc_auc"] + 0.3 * m["f1"] + 0.3 * m["sensitivity"]
        scores[key] = composite

    best_key = max(scores, key=scores.get)
    logger.info(
        f"Best model: {valid_results[best_key]['name']} "
        f"(composite={scores[best_key]:.3f})"
    )

    return best_key, valid_results[best_key]


# ─── Artifact Persistence ────────────────────────────────────────────────────

def save_model_artifacts(
    model,
    scaler,
    feature_names: List[str],
    metrics: Dict,
    model_key: str,
    version: str = None,
    dataset_report: Optional[Dict] = None,
) -> Path:
    """Save model, scaler, and metadata to disk."""
    if version is None:
        version = DEFAULT_MODEL_VERSION

    artifact_dir = MODEL_DIR / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = artifact_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Save metadata
    metadata = {
        "model_name": "pd_voice_classifier",
        "model_type": model_key,
        "version": version,
        "feature_schema_version": "1.0",
        "preprocessing_version": "1.0",
        "dataset": {
            "name": "UCI Oxford Parkinson's Disease Detection",
            "source": "https://archive.ics.uci.edu/ml/datasets/parkinsons",
            "citation": (
                "Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM (2007) "
                "'Exploiting Nonlinear Recurrence and Fractal Scaling Properties "
                "for Voice Disorder Detection', BioMedical Engineering OnLine."
            ),
            "n_subjects": 31,
            "n_recordings": 195,
            "n_features": len(feature_names),
        },
        "features": feature_names,
        "target": "status",
        "target_encoding": {"0": "healthy", "1": "parkinson"},
        "training_date": datetime.now(timezone.utc).isoformat(),
        "validation_protocol": "subject-aware group shuffle split (80/20)",
        "metrics": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report_dict")},
        "intended_use": (
            "Research and screening support tool for voice-based Parkinson's "
            "disease indicator detection. Not for clinical diagnosis."
        ),
        "known_limitations": [
            "Trained on a small dataset of 31 subjects (195 recordings).",
            "Only uses voice biomedical features, not clinical symptoms.",
            "Performance may not generalize to different populations or recording equipment.",
            "Cannot diagnose Parkinson's disease — only identifies voice patterns.",
            "Subject-level performance may vary significantly.",
            "Not validated in a clinical trial.",
        ],
        "status": "validated",
    }

    if dataset_report:
        metadata["dataset_quality"] = {
            "total_missing": dataset_report.get("total_missing", 0),
            "inconsistent_subjects": dataset_report.get("inconsistent_subjects", 0),
        }

    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved model artifacts to {artifact_dir}")
    return artifact_dir


def load_model_artifacts(version: str = None) -> Tuple:
    """Load model, scaler, and metadata from disk."""
    if version is None:
        version = DEFAULT_MODEL_VERSION

    artifact_dir = MODEL_DIR / version

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Model artifacts not found: {artifact_dir}")

    model = joblib.load(artifact_dir / "model.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(f"Loaded model artifacts from {artifact_dir}")
    return model, scaler, metadata


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(
    version: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Run the complete training pipeline:
    1. Download/load dataset
    2. Inspect quality
    3. Subject-aware split
    4. Preprocess (scaler fit on train only)
    5. Train all models
    6. Evaluate & select best
    7. Save artifacts
    """
    if version is None:
        version = DEFAULT_MODEL_VERSION

    logger.info("=" * 60)
    logger.info(f"PD-Aware Training Pipeline — version {version}")
    logger.info("=" * 60)

    # Step 1: Load dataset
    filepath = download_dataset()
    df = load_dataset(filepath)

    # Step 2: Quality inspection
    report = inspect_dataset(df)
    logger.info(f"Dataset: {report['n_rows']} rows, {report['n_subjects']} subjects")
    logger.info(f"Target distribution: {report['target_distribution']}")
    logger.info(f"Missing values: {report['total_missing']}")

    # Step 3: Subject-aware split
    train_df, test_df = subject_aware_split(df, test_size=test_size, random_state=random_state)

    # Step 4: Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)

    # Step 5: Train all models
    results = train_all_models(X_train, y_train, X_test, y_test, feature_names)

    # Step 6: Select best
    best_key, best_result = select_best_model(results)

    # Step 7: Generate evaluation report
    eval_report = generate_evaluation_report(results, feature_names)

    # Step 8: Save artifacts
    artifact_dir = save_model_artifacts(
        model=best_result["model"],
        scaler=scaler,
        feature_names=feature_names,
        metrics=best_result["metrics"],
        model_key=best_key,
        version=version,
        dataset_report=report,
    )

    pipeline_result = {
        "version": version,
        "dataset_report": report,
        "all_results": {
            k: {key: val for key, val in v.items() if key != "model"}
            for k, v in results.items()
        },
        "best_model": best_key,
        "best_model_name": best_result["name"],
        "best_metrics": best_result["metrics"],
        "artifact_dir": str(artifact_dir),
        "feature_names": feature_names,
        "evaluation_report": eval_report,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Best model: {best_result['name']}")
    logger.info("=" * 60)

    return pipeline_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_full_pipeline()
    print(json.dumps(
        {k: v for k, v in result.items() if k != "evaluation_report"},
        indent=2, default=str,
    ))
