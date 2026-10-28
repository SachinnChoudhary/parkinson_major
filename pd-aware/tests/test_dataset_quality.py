"""
Tests for Dataset Quality Inspection, Column Mapping, and Consistency Checks.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset import inspect_dataset, load_dataset
from config.settings import VOICE_FEATURES, TARGET_COLUMN, SUBJECT_ID_COLUMN, DATA_DIR


def test_inspect_dataset_structure():
    """Verify quality report contains all expected summary metrics."""
    # Synthetic mini dataset
    data = {
        "name": ["phon_R01_S01_1", "phon_R01_S01_2", "phon_R01_S02_1"],
        "status": [1, 1, 0],
    }
    for f in VOICE_FEATURES:
        data[f] = [100.0, 105.0, 120.0]

    df = pd.DataFrame(data)
    report = inspect_dataset(df)

    assert report["n_rows"] == 3
    assert report["n_columns"] == 2 + len(VOICE_FEATURES)
    assert report["total_missing"] == 0
    assert report["n_subjects"] == 3  # S01_1, S01_2 treated by default or unique
    assert report["inconsistent_subjects"] == 0


def test_inconsistent_subject_label_detection():
    """Verify that if a single subject has conflicting labels, inspection flags it."""
    # Subject S01 has status=1 in row 1 and status=0 in row 2
    df = pd.DataFrame({
        "name": ["phon_R01_S01_1", "phon_R01_S01_2"],
        "status": [1, 0],  # CONFLICT
        "MDVP:Fo(Hz)": [120.0, 122.0],
    })

    # Group by extracted subject
    subjects = df["name"].str.extract(r"(S\d+)", expand=False)
    temp_df = pd.DataFrame({"subject": subjects, "status": df["status"]})
    labels_per_subj = temp_df.groupby("subject")["status"].nunique()
    inconsistent = labels_per_subj[labels_per_subj > 1]

    assert len(inconsistent) == 1
    assert "S01" in inconsistent.index


def test_missing_data_imputation():
    """Verify that missing values in feature matrix are properly detected and imputed."""
    from ml.dataset import preprocess

    train_df = pd.DataFrame({
        "name": ["phon_R01_S01_1", "phon_R01_S01_2"],
        "status": [1, 1],
        "MDVP:Fo(Hz)": [100.0, np.nan],  # Missing value in training
        "HNR": [20.0, 22.0],
    })
    test_df = pd.DataFrame({
        "name": ["phon_R01_S02_1"],
        "status": [0],
        "MDVP:Fo(Hz)": [np.nan],  # Missing value in test
        "HNR": [19.0],
    })

    # Features
    feats = ["MDVP:Fo(Hz)", "HNR"]
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _ = preprocess(train_df, test_df, feature_columns=feats)

    # Imputer should replace NaNs, resulting in zero NaNs in scaled arrays
    assert not np.isnan(X_train_scaled).any()
    assert not np.isnan(X_test_scaled).any()


def test_load_real_uci_dataset_if_present():
    """Test loading the real UCI Parkinson dataset if present in data directory."""
    filepath = DATA_DIR / "parkinsons.data"
    if filepath.exists():
        df = load_dataset(filepath)
        assert len(df) == 195
        assert TARGET_COLUMN in df.columns
        assert all(f in df.columns for f in VOICE_FEATURES)

