"""Deterministic cleaning and transformation pipeline for the Bank Marketing dataset.

Responsibilities
----------------
* Load the raw dataset from ``data/raw/bank-additional-full.csv`` (preferred) or
  ``data/raw/dataset.csv`` (fallback), auto-detecting the ``;`` / ``,`` delimiter.
* Remove exact duplicate rows.
* Standardize string / categorical columns (strip whitespace).
* Treat ``"unknown"`` values as explicit missingness (converted to ``NaN``;
  downstream modelling keeps this information as its own ``"missing"`` category
  instead of dropping rows or inventing values).
* Handle ``pdays == 999`` (client was not previously contacted) thoughtfully:
  a binary ``was_previously_contacted`` indicator is created and ``999`` is
  replaced with ``NaN`` rather than being treated as a literal day count.
* Drop ``duration`` from predictive features to prevent post-call data leakage
  (the call duration is unknown before a call is performed and directly leaks
  the outcome; see Moro et al., 2014).

The cleaned frame is persisted to ``data/processed/bank_clean.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PRIMARY_PATH = RAW_DIR / "bank-additional-full.csv"
FALLBACK_PATH = RAW_DIR / "dataset.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_PATH = PROCESSED_DIR / "bank_clean.csv"

TARGET = "y"
LEAKAGE_COLUMN = "duration"

# Raw column groups as documented in bank-additional-names.txt (Moro et al., 2014).
NUMERIC_COLUMNS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_COLUMNS = [
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

# Value used in ``pdays`` when a client was never contacted in a previous campaign.
PDAYS_NEVER_CONTACTED = 999


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
    """Detect whether the CSV uses ``;`` or ``,`` as its delimiter."""
    semicolons = commas = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(sample_lines):
            line = handle.readline()
            if not line:
                break
            semicolons += line.count(";")
            commas += line.count(",")
    delimiter = ";" if semicolons >= commas else ","
    logger.info("Detected delimiter %r for %s", delimiter, path)
    return delimiter


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw bank marketing CSV.

    Prefers ``data/raw/bank-additional-full.csv`` and falls back to
    ``data/raw/dataset.csv`` with automatic delimiter detection.
    """
    candidates = [path] if path is not None else [PRIMARY_PATH, FALLBACK_PATH]
    for candidate in candidates:
        if candidate.exists():
            delimiter = detect_delimiter(candidate)
            df = pd.read_csv(candidate, sep=delimiter)
            logger.info(
                "Loaded raw data from %s: %d rows x %d columns",
                candidate,
                df.shape[0],
                df.shape[1],
            )
            return df
    raise FileNotFoundError(
        f"Neither {PRIMARY_PATH} nor {FALLBACK_PATH} was found in {RAW_DIR}"
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic cleaning pipeline to a raw DataFrame.

    Steps
    -----
    1. Standardize object columns (strip whitespace).
    2. Convert ``"unknown"`` entries in categorical columns to ``NaN``.
    3. Remove exact duplicate rows.
    4. Drop ``duration`` (post-call leakage).
    5. Create ``was_previously_contacted`` from ``pdays``; replace ``999`` with
       ``NaN`` so it is not modelled as a literal number of days.
    """
    df = df.copy()

    # --- 1. Standardize string / categorical columns -----------------------
    for col in df.columns:
        if df[col].dtype == object:
            before_nans = df[col].isna().sum()
            df[col] = df[col].astype(str).str.strip()
            # Restore genuine NaNs that astype(str) turned into the string "nan".
            df[col] = df[col].replace({"nan": pd.NA})
            logger.debug(
                "Standardized column %r (was %d missing)", col, int(before_nans)
            )

    # --- 2. Treat "unknown" as explicit missingness ------------------------
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            n_unknown = int((df[col] == "unknown").sum())
            if n_unknown:
                df[col] = df[col].replace({"unknown": pd.NA})
                logger.info('Converted %d "unknown" values to NaN in column %r', n_unknown, col)

    # --- 3. Drop duration (post-call data leakage) BEFORE deduplication ---
    # Two records may describe the same client profile with different call
    # durations; once duration is removed they become exact duplicates, which
    # must not be split across train/test (client-level leakage).
    if LEAKAGE_COLUMN in df.columns:
        df = df.drop(columns=[LEAKAGE_COLUMN])
        logger.info("Dropped leakage column %r", LEAKAGE_COLUMN)

    # --- 4. Handle pdays == 999 (not previously contacted) -----------------
    if "pdays" in df.columns:
        df["pdays"] = df["pdays"].astype("Int64")
        was_contacted = df["pdays"] != PDAYS_NEVER_CONTACTED
        df["was_previously_contacted"] = was_contacted.astype("Int64")
        df.loc[~was_contacted, "pdays"] = pd.NA
        n_never = int((~was_contacted).sum())
        logger.info(
            "Created was_previously_contacted (%d clients never previously contacted; pdays=999 set to NaN)",
            n_never,
        )

    # --- 5. Remove exact duplicates (after leakage column is gone) ---------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        logger.info("Removed %d exact duplicate row(s)", n_removed)

    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    """Persist the cleaned frame to CSV and return the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s (%d rows x %d columns)", path, df.shape[0], df.shape[1])
    return path


def run(path: Path | None = None) -> pd.DataFrame:
    """End-to-end cleaning entry point: load, clean, persist."""
    df_raw = load_raw(path)
    df_clean = clean_dataframe(df_raw)
    save_processed(df_clean)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    )
    run()
