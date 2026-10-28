"""
Tests for Natural Language Symptom Extraction, Time Extraction, and Questionnaire Validation.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.symptom_extractor import extract_symptoms_keyword, extract_time_info
from config.settings import MOTOR_SYMPTOMS, NON_MOTOR_SYMPTOMS, SEVERITY_LEVELS


def test_keyword_symptom_extraction_motor():
    """Verify keyword extraction identifies motor symptoms in natural text."""
    sample_text = (
        "I have noticed shaking in my right hand when sitting relaxed. "
        "My handwriting has become very small and my legs feel stiff."
    )
    extracted = extract_symptoms_keyword(sample_text)
    extracted_names = [s["name"] for s in extracted]

    assert any("Tremor" in name for name in extracted_names)
    assert any("handwriting" in name.lower() for name in extracted_names)
    assert any("stiffness" in name.lower() or "rigidity" in name.lower() for name in extracted_names)


def test_keyword_symptom_extraction_non_motor():
    """Verify keyword extraction identifies non-motor symptoms."""
    sample_text = "I have lost my sense of smell and have terrible insomnia and constipation."
    extracted = extract_symptoms_keyword(sample_text)
    extracted_names = [s["name"] for s in extracted]

    assert any("smell" in name.lower() for name in extracted_names)
    assert any("constipation" in name.lower() for name in extracted_names)


def test_time_information_extraction():
    """Verify temporal information is extracted from narrative descriptions."""
    text = "These tremors started 6 months ago, and over the past 2 years I felt progressively slower."
    time_info = extract_time_info(text)

    assert "duration" in time_info or "onset" in time_info
    all_time_tokens = " ".join([item for sub in time_info.values() for item in sub])
    assert "month" in all_time_tokens or "year" in all_time_tokens


def test_questionnaire_definitions():
    """Verify structured symptom questionnaire definitions and severity scales."""
    assert len(MOTOR_SYMPTOMS) >= 5
    assert len(NON_MOTOR_SYMPTOMS) >= 5
    assert len(SEVERITY_LEVELS) >= 3

    # Check motor symptom tuples have (id, name, description)
    for item in MOTOR_SYMPTOMS:
        assert len(item) == 3
        assert isinstance(item[0], str)
        assert isinstance(item[1], str)

    # Check severity levels include none and moderate/severe
    severity_keys = [s[0] for s in SEVERITY_LEVELS]
    assert "none" in severity_keys
    assert "moderate" in severity_keys
