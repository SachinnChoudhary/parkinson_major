"""
UCI Oxford Parkinson's Disease Dataset loader and preprocessing.

Dataset: https://archive.ics.uci.edu/ml/datasets/parkinsons
Paper: Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM (2007)
       'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for
       Voice Disorder Detection', BioMedical Engineering OnLine.

Contains 195 voice recordings from 31 subjects (23 with PD, 8 healthy).
Each row is a voice recording; multiple recordings per subject.

CRITICAL: We must split by SUBJECT (name column), not by row, to prevent
data leakage. Recordings from the same person must not appear in both
train and test sets.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config.settings import (
    DATA_DIR, UCI_DATASET_URL, VOICE_FEATURES, TARGET_COLUMN, SUBJECT_ID_COLUMN,
)

logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Download the UCI Parkinson's dataset if not already present."""
    filepath = DATA_DIR / "parkinsons.data"

    if filepath.exists() and not force:
        logger.info(f"Dataset already exists at {filepath}")
        return filepath

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset from {UCI_DATASET_URL}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(UCI_DATASET_URL, filepath)
        logger.info(f"Dataset saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    return filepath


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the UCI Parkinson's dataset."""
    if filepath is None:
        filepath = DATA_DIR / "parkinsons.data"

    if not filepath.exists():
        filepath = download_dataset()

    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict:
    """Perform quality checks on the dataset."""
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "n_subjects": df[SUBJECT_ID_COLUMN].nunique(),
        "recordings_per_subject": df.groupby(SUBJECT_ID_COLUMN).size().describe().to_dict(),
    }

    # Subject-level label check: ensure each subject has only one label
    subject_labels = df.groupby(SUBJECT_ID_COLUMN)[TARGET_COLUMN].nunique()
    inconsistent = subject_labels[subject_labels > 1]
    report["inconsistent_subjects"] = len(inconsistent)
    report["inconsistent_subject_names"] = list(inconsistent.index) if len(inconsistent) > 0 else []

    # Feature statistics
    feature_cols = [c for c in VOICE_FEATURES if c in df.columns]
    report["feature_stats"] = df[feature_cols].describe().to_dict()
    report["available_features"] = feature_cols
    report["missing_features"] = [c for c in VOICE_FEATURES if c not in df.columns]

    return report


def extract_subject_id(name_column: pd.Series) -> pd.Series:
    """
    Extract subject ID from the 'name' column.
    Format is like 'phon_R01_S01_1' -> 'S01' is the subject.
    """
    return name_column.str.extract(r"(S\d+)", expand=False)


def subject_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by subject to prevent data leakage.

    All recordings from a given subject go entirely into train OR test,
    never split across both.
    """
    subjects = extract_subject_id(df[SUBJECT_ID_COLUMN])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df[TARGET_COLUMN], groups=subjects))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Verify no subject overlap
    train_subjects = set(extract_subject_id(train_df[SUBJECT_ID_COLUMN]))
    test_subjects = set(extract_subject_id(test_df[SUBJECT_ID_COLUMN]))
    overlap = train_subjects & test_subjects

    if overlap:
        raise ValueError(
            f"Subject leakage detected! Overlapping subjects: {overlap}"
        )

    logger.info(
        f"Subject-aware split: train={len(train_df)} rows ({len(train_subjects)} subjects), "
        f"test={len(test_df)} rows ({len(test_subjects)} subjects)"
    )

    return train_df, test_df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Preprocess the dataset: extract features, scale.

    CRITICAL: Scaler is fit ONLY on training data to prevent preprocessing leakage.
    """
    if feature_columns is None:
        feature_columns = [c for c in VOICE_FEATURES if c in train_df.columns]

    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values
    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # Check for missing values
    train_missing = np.isnan(X_train).sum()
    test_missing = np.isnan(X_test).sum()
    if train_missing > 0 or test_missing > 0:
        logger.warning(f"Missing values found: train={train_missing}, test={test_missing}")
        # Impute with column median from training set
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    # Scale features — fit ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        f"Preprocessed: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}, "
        f"y_train distribution={dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
