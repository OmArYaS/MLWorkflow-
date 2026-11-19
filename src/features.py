"""Feature engineering and preprocessing pipeline for the Bank Marketing dataset.

Builds a scikit-learn ``ColumnTransformer`` + ``Pipeline`` that is fitted
*strictly on training data* (no leakage):

* Numeric features: median imputation (fitted on train only) + standard scaling.
  ``pdays == 999`` is already encoded as NaN by the cleaning step plus a
  ``was_previously_contacted`` indicator, so median imputation does not invent
  fake day counts for never-contacted clients.
* Categorical features: missingness (including former ``"unknown"`` values) is
  imputed with an explicit ``"missing"`` category, then one-hot encoded. This
  preserves the information that a value was not recorded instead of dropping
  rows or fabricating a mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import PROCESSED_PATH

logger = logging.getLogger(__name__)

TARGET = "y"

# Numeric features used for modelling (``duration`` is deliberately excluded;
# see cleaning.LEAKAGE_COLUMN).
NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "was_previously_contacted",
]

# Categorical features used for modelling.
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned processed frame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run `python -m src.cleaning` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded processed data from %s: %d rows x %d columns", path, df.shape[0], df.shape[1])
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into feature frame ``X`` and binary target ``y`` (1 = subscribed)."""
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[ALL_FEATURES].copy()
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing ``ColumnTransformer``.

    Notes
    -----
    Must be fitted on training data only (handled by the caller / pipeline).
    Categorical missingness is encoded as an explicit ``"missing"`` category so
    the model can learn from the fact that a value was not recorded.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    try:
        # Human-readable DataFrame output (scikit-learn >= 1.2).
        preprocessor.set_output(transform="pandas")
    except Exception:  # pragma: no cover - version fallback
        logger.debug("set_output(transform='pandas') not supported by this scikit-learn version")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names of a fitted preprocessor."""
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform ``X`` with a fitted preprocessor, returning a labelled DataFrame."""
    transformed = preprocessor.transform(X)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(
            transformed, columns=get_feature_names(preprocessor), index=X.index
        )
    return transformed
