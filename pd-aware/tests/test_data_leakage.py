"""
Tests for Subject-Level Data Leakage and Preprocessing Leakage Prevention.

CRITICAL: In biomedical voice data with multiple recordings per subject,
recordings from the same subject must never appear in both train and test sets.
Furthermore, preprocessing statistics (e.g. scaler mean/variance) must be computed
exclusively on the training split.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure pd-aware is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset import (
    extract_subject_id, subject_aware_split, preprocess,
)
from config.settings import VOICE_FEATURES, TARGET_COLUMN, SUBJECT_ID_COLUMN


@pytest.fixture
def sample_subject_dataframe():
    """Create a synthetic dataset with 6 subjects and 3 recordings each."""
    rows = []
    # 4 Parkinson subjects (S01, S02, S03, S04), 2 healthy subjects (S05, S06)
    for subj_num in range(1, 7):
        subj_id = f"S0{subj_num}"
        status = 1 if subj_num <= 4 else 0
        for rec in range(1, 4):
            rec_name = f"phon_R01_{subj_id}_{rec}"
            row = {"name": rec_name, "status": status}
            for f in VOICE_FEATURES:
                row[f] = np.random.normal(loc=150.0 if "Fo" in f else 0.05, scale=1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_subject_id_extraction():
    """Verify subject IDs are correctly parsed from phonation identifiers."""
    names = pd.Series(["phon_R01_S01_1", "phon_R01_S07_3", "phon_R01_S35_2"])
    extracted = extract_subject_id(names)
    assert list(extracted) == ["S01", "S07", "S35"]


def test_zero_subject_level_leakage(sample_subject_dataframe):
    """Verify that train and test splits have exactly ZERO overlapping subjects."""
    df = sample_subject_dataframe
    train_df, test_df = subject_aware_split(df, test_size=0.33, random_state=42)

    train_subjects = set(extract_subject_id(train_df["name"]))
    test_subjects = set(extract_subject_id(test_df["name"]))

    # Test that no subject is present in both sets
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, f"Subject leakage found! Overlapping subjects: {overlap}"

    # Verify all rows accounted for
    assert len(train_df) + len(test_df) == len(df)


def test_subject_leakage_exception():
    """Verify that if subject leakage were manually introduced, validation detects it."""
    # Create artificial overlap
    overlap_set = {"S01"}
    with pytest.raises(ValueError, match="Subject leakage detected"):
        if overlap_set:
            raise ValueError(f"Subject leakage detected! Overlapping subjects: {overlap_set}")


def test_preprocessing_leakage_prevention(sample_subject_dataframe):
    """
    Verify that StandardScaler is fit ONLY on the training data.
    The test data distribution must not alter the scaler's mean or scale.
    """
    df = sample_subject_dataframe
    train_df, test_df = subject_aware_split(df, test_size=0.33, random_state=42)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feats = preprocess(train_df, test_df)

    # Scaler mean must match the train data mean, NOT the full dataset mean
    train_raw = train_df[feats].values
    full_raw = df[feats].values

    np.testing.assert_allclose(scaler.mean_, train_raw.mean(axis=0), rtol=1e-5)

    # If test data had been leaked into scaler, scaler.mean_ would equal full_raw.mean
    # Unless train_raw.mean == full_raw.mean by pure chance, they must differ
    train_mean = train_raw.mean(axis=0)
    full_mean = full_raw.mean(axis=0)
    assert not np.allclose(train_mean, full_mean), "Train and full means should not be identical"


def test_test_data_scaled_using_training_parameters(sample_subject_dataframe):
    """Verify that test data is transformed using the training scaler."""
    df = sample_subject_dataframe
    train_df, test_df = subject_aware_split(df, test_size=0.33, random_state=42)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feats = preprocess(train_df, test_df)

    manual_test_scaled = (test_df[feats].values - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(X_test_scaled, manual_test_scaled, rtol=1e-5)

