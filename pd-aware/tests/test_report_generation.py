"""
Tests for PDF Screening Report Generation and Medical Disclaimer Enforcement.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.report_generator import generate_screening_report
from config.settings import MEDICAL_DISCLAIMER


def test_generate_report_symptoms_only():
    """Verify PDF report generation when only symptom questionnaire was completed."""
    screening_data = {
        "age_range": "60-69",
        "symptom_duration": "1-2 years",
        "existing_diagnosis": "No",
        "family_history": "None known",
        "natural_language_input": "Mild hand shaking noticed when writing.",
        "user_reviewed_symptoms": [{"name": "Tremor", "severity": "Mild"}],
        "questionnaire_responses": {"tremor_rest": "Mild", "bradykinesia": "Not present"},
    }

    pdf_bytes = generate_screening_report(
        report_id="TEST-SYM-01",
        screening_data=screening_data,
        prediction_data=None,
        explanation_data=None,
        include_symptoms=True,
        include_prediction=False,
        include_explanation=False,
    )

    assert type(pdf_bytes) is bytes
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_report_with_ml_prediction_and_shap():
    """Verify PDF report generation with full ML output and SHAP explanations."""
    screening_data = {
        "age_range": "70-79",
        "symptom_duration": "6-12 months",
        "user_reviewed_symptoms": [{"name": "Muscle stiffness", "severity": "Moderate"}],
    }

    prediction_data = {
        "success": True,
        "prediction_label": 1,
        "prediction_category": "parkinson",
        "prediction_score": 0.885,
        "model_version": "v1.0.0",
        "model_type": "gradient_boosting",
    }

    explanation_data = {
        "success": True,
        "top_contributors": [
            {"feature": "MDVP:Fo(Hz)", "description": "Average fundamental frequency", "direction": "increases risk", "magnitude": 0.35},
            {"feature": "PPE", "description": "Pitch period entropy", "direction": "increases risk", "magnitude": 0.28},
        ],
    }

    pdf_bytes = generate_screening_report(
        report_id="TEST-FULL-02",
        screening_data=screening_data,
        prediction_data=prediction_data,
        explanation_data=explanation_data,
        include_symptoms=True,
        include_prediction=True,
        include_explanation=True,
    )

    assert type(pdf_bytes) is bytes
    assert len(pdf_bytes) > 1500
    assert pdf_bytes.startswith(b"%PDF")


def test_report_contains_non_diagnostic_language():
    """Verify disclaimer text is prominent in settings."""
    assert "not diagnose" in MEDICAL_DISCLAIMER.lower()
    assert "screening information" in MEDICAL_DISCLAIMER.lower()

