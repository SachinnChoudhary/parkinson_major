"""
Tests for Model Training, Evaluation Metrics, Inference Service, and Explainability.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import VOICE_FEATURES
from ml.train import get_model_configs, train_single_model
from ml.evaluate import evaluate_model
from ml.inference import InferenceService, get_inference_service
from ml.explainability import explain_prediction, get_global_feature_importance


@pytest.fixture
def dummy_train_test_data():
    """Create reproducible synthetic binary classification data."""
    rng = np.random.RandomState(42)
    n_train, n_test = 60, 20
    n_features = len(VOICE_FEATURES)

    X_train = rng.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = rng.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    return X_train, y_train, X_test, y_test


def test_candidate_model_definitions():
    """Verify that all core candidate models are configured properly."""
    configs = get_model_configs()
    expected = ["logistic_regression", "svm", "random_forest", "extra_trees", "gradient_boosting", "xgboost"]
    for key in expected:
        assert key in configs
        assert "model" in configs[key]
        assert "name" in configs[key]


def test_model_training_and_evaluation(dummy_train_test_data):
    """Verify training a model computes expected classification metrics."""
    X_train, y_train, X_test, y_test = dummy_train_test_data
    model, t_time = train_single_model("logistic_regression", X_train, y_train, VOICE_FEATURES)

    assert model is not None
    assert t_time >= 0

    metrics = evaluate_model(model, X_test, y_test, model_name="Logistic Regression")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_inference_service_feature_validation():
    """Verify feature validation handles complete, missing, and invalid inputs."""
    service = get_inference_service()
    if not service.is_loaded:
        pytest.skip("Model not yet trained/loaded")

    # Complete valid features
    valid_features = {f: 0.1 for f in service.expected_features}
    is_valid, issues = service.validate_features(valid_features)
    assert is_valid
    assert len(issues) == 0

    # Missing features
    incomplete = {"MDVP:Fo(Hz)": 150.0}
    is_valid, issues = service.validate_features(incomplete)
    assert not is_valid
    assert any("Missing features" in i for i in issues)

    # Invalid non-numeric input
    invalid_data = {f: 0.1 for f in service.expected_features}
    invalid_data["MDVP:Fo(Hz)"] = "invalid_string"
    is_valid, issues = service.validate_features(invalid_data)
    assert not is_valid
    assert any("non-numeric" in i for i in issues)


def test_inference_prediction_response_structure():
    """Verify structured response schema from InferenceService."""
    service = get_inference_service()
    if not service.is_loaded:
        pytest.skip("Model not yet trained/loaded")

    features = {f: 0.1 for f in service.expected_features}
    result = service.predict(features)

    assert result["success"] is True
    assert result["prediction_label"] in (0, 1)
    assert result["prediction_category"] in ("healthy", "parkinson")
    assert "prediction_score" in result
    assert "inference_time_ms" in result
    assert "model_version" in result
    assert "input_features" in result
    assert "limitations" in result


def test_model_unavailable_handling():
    """Verify graceful handling when an invalid model version is requested."""
    unloaded_service = InferenceService(version="v999.nonexistent")
    res = unloaded_service.predict({"MDVP:Fo(Hz)": 100.0})

    assert res["success"] is False
    assert "Model not loaded" in res["error"]


def test_explainability_generation():
    """Verify SHAP explanation generation for a single prediction."""
    service = get_inference_service()
    if not service.is_loaded:
        pytest.skip("Model not yet trained/loaded")

    raw_vals = np.array([[0.1 for _ in service.expected_features]])
    exp = explain_prediction(service.model, service.scaler, raw_vals, service.expected_features)

    assert exp["success"] is True
    assert "contributions" in exp
    assert len(exp["contributions"]) == len(service.expected_features)
    assert "top_contributors" in exp
    assert len(exp["top_contributors"]) <= 5


def test_global_feature_importance():
    """Verify global feature importance calculation."""
    service = get_inference_service()
    if not service.is_loaded:
        pytest.skip("Model not yet trained/loaded")

    glob = get_global_feature_importance(service.model, service.expected_features)
    assert glob["success"] is True
    assert "importances" in glob
    assert len(glob["importances"]) == len(service.expected_features)

