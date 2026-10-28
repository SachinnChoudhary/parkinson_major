"""
End-to-End Integration Test: Screening Journey to PDF Report.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import VOICE_FEATURES
from database.engine import init_db, get_session
from database.models import Screening, Prediction, ScreeningStatus
from nlp.symptom_extractor import extract_symptoms_keyword
from ml.inference import get_inference_service
from ml.explainability import explain_prediction
from components.report_generator import generate_screening_report


def test_complete_screening_to_report_journey():
    """
    Test the full end-to-end screening journey:
    1. Informed consent
    2. Patient demographic and history inputs
    3. Natural language symptom description and NLP extraction
    4. Structured symptom questionnaire
    5. Voice acoustic data processing
    6. Machine learning inference and SHAP explainability
    7. Database persistence of screening and prediction records
    8. Downloadable PDF clinical report generation
    """
    init_db()
    session_id = str(uuid.uuid4())

    # Step 1: Consent
    consent_given = True
    consent_timestamp = datetime.now(timezone.utc)

    # Step 2: Patient Info
    age_range = "60-69"
    symptom_duration = "6-12 months"
    existing_diagnosis = "Under investigation"
    family_history = "Yes — immediate family"

    # Step 3: Natural language description & NLP
    nl_text = "I have noticed a rest tremor in my right hand and walking feels slower."
    extracted_symptoms = extract_symptoms_keyword(nl_text)
    user_reviewed_symptoms = extracted_symptoms  # User accepts extracted symptoms

    # Step 4: Questionnaire
    questionnaire = {
        "tremor_rest": "Mild – I notice it but it doesn't affect daily activities",
        "bradykinesia": "Mild – I notice it but it doesn't affect daily activities",
        "smell": "Mild – I notice it but it doesn't affect daily activities",
    }

    # Step 5: Voice acoustic features
    service = get_inference_service()
    if not service.is_loaded:
        pytest.skip("Inference model not loaded")

    # Generate synthetic voice features conforming to feature schema
    voice_features = {f: 0.1 for f in service.expected_features}
    voice_features["MDVP:Fo(Hz)"] = 120.0
    voice_features["MDVP:Jitter(%)"] = 0.008
    voice_features["MDVP:Shimmer"] = 0.045
    voice_features["HNR"] = 21.0

    # Step 6: ML Inference & SHAP explanation
    prediction_result = service.predict(voice_features)
    assert prediction_result["success"] is True

    raw_features_array = np.array([[voice_features[f] for f in service.expected_features]])
    explanation_result = explain_prediction(
        service.model, service.scaler, raw_features_array, service.expected_features
    )
    assert explanation_result["success"] is True

    # Step 7: Database persistence
    db = get_session()
    screening_record = Screening(
        session_id=session_id,
        status=ScreeningStatus.COMPLETED,
        age_range=age_range,
        symptom_duration=symptom_duration,
        existing_diagnosis=existing_diagnosis,
        family_history=family_history,
        natural_language_input=nl_text,
        extracted_symptoms=extracted_symptoms,
        user_reviewed_symptoms=user_reviewed_symptoms,
        questionnaire_responses=questionnaire,
        voice_features=voice_features,
        consent_given=consent_given,
        consent_timestamp=consent_timestamp,
        completed_at=datetime.now(timezone.utc),
    )
    db.add(screening_record)
    db.commit()

    prediction_record = Prediction(
        screening_id=screening_record.id,
        model_version=prediction_result["model_version"],
        model_type=prediction_result["model_type"],
        feature_schema_version=prediction_result["feature_schema_version"],
        preprocessing_version=prediction_result["preprocessing_version"],
        prediction_label=prediction_result["prediction_label"],
        prediction_score=prediction_result["prediction_score"],
        prediction_category=prediction_result["prediction_category"],
        input_features=voice_features,
        feature_names=service.expected_features,
        shap_values=explanation_result.get("shap_values"),
        inference_time_ms=prediction_result["inference_time_ms"],
    )
    db.add(prediction_record)
    db.commit()

    # Verify database persistence
    saved_screening = db.query(Screening).filter(Screening.session_id == session_id).first()
    assert saved_screening is not None
    assert saved_screening.prediction is not None
    assert saved_screening.prediction.prediction_label in (0, 1)

    # Step 8: PDF Report Generation
    report_id = f"PDA-{session_id[:8].upper()}"
    pdf_bytes = generate_screening_report(
        report_id=report_id,
        screening_data={
            "age_range": age_range,
            "symptom_duration": symptom_duration,
            "existing_diagnosis": existing_diagnosis,
            "family_history": family_history,
            "natural_language_input": nl_text,
            "user_reviewed_symptoms": user_reviewed_symptoms,
            "questionnaire_responses": questionnaire,
        },
        prediction_data=prediction_result,
        explanation_data=explanation_result,
    )

    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")
    db.close()

