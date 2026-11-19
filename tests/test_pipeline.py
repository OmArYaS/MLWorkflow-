"""Smoke tests for the Bank Marketing data-cleaning / modelling pipeline.

Run with:  pytest tests/test_pipeline.py
"""

import sys
from pathlib import Path

# Ensure the repository root is importable when pytest is invoked from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    LEAKAGE_COLUMN,
    PDAYS_NEVER_CONTACTED,
    clean_dataframe,
    load_raw,
)
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    get_feature_names,
    split_xy,
)
from src.train_evaluate import MODEL_PATH, run as run_training


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_dataframe(raw_df)


def test_load_data_returns_expected_shape(raw_df: pd.DataFrame) -> None:
    assert isinstance(raw_df, pd.DataFrame)
    assert not raw_df.empty
    assert "y" in raw_df.columns


def test_duration_dropped_to_prevent_leakage(clean_df: pd.DataFrame) -> None:
    assert LEAKAGE_COLUMN not in clean_df.columns


def test_unknown_values_are_handled(clean_df: pd.DataFrame) -> None:
    """No literal ``"unknown"`` strings should remain in categorical columns."""
    for col in CATEGORICAL_FEATURES:
        if col in clean_df.columns:
            unknown_mask = clean_df[col].astype(str).str.strip().eq("unknown")
            assert unknown_mask.sum() == 0, f'{unknown_mask.sum()} "unknown" values remain in {col!r}'


def test_pdays_999_is_not_treated_as_literal_days(clean_df: pd.DataFrame) -> None:
    assert "was_previously_contacted" in clean_df.columns
    pdays = clean_df["pdays"].dropna()
    assert (pdays != PDAYS_NEVER_CONTACTED).all(), "pdays==999 should have been replaced with NaN"
    assert clean_df["was_previously_contacted"].isin([0, 1, pd.NA]).all()


def test_no_exact_duplicates_after_cleaning(clean_df: pd.DataFrame) -> None:
    assert clean_df.duplicated().sum() == 0


def test_preprocessing_has_no_nans(clean_df: pd.DataFrame) -> None:
    X, y = split_xy(clean_df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    arr = np.asarray(X_transformed)
    assert not pd.isna(arr).any(), "Preprocessed features must contain no NaNs"


def test_feature_names_match_transformed_shape(clean_df: pd.DataFrame) -> None:
    X, _ = split_xy(clean_df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)
    assert X_transformed.shape[1] == len(names)


def test_model_inference_shape(clean_df: pd.DataFrame) -> None:
    """The (best) pipeline must produce predictions matching the test-set shape."""
    X, y = split_xy(clean_df)
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if MODEL_PATH.exists():
        import joblib

        pipeline = joblib.load(MODEL_PATH)
    else:
        # Smoke path: train a quick baseline so the test remains self-contained.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=42
                    ),
                ),
            ]
        )
        _, X_train, _, y_train = train_test_split(
            X, y, test_size=0.8, stratify=y, random_state=42
        )
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    assert y_pred.shape == y_test.shape
    assert y_prob.shape == (len(y_test), 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
