"""Data profiling for the Bank Marketing dataset.

Inspects shapes, dtypes, ``"unknown"`` counts, duplicates, cardinality, class
imbalance and outliers (IQR method), and exports a data-quality summary report
as JSON and Markdown to ``artifacts/reports/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_string_dtype

from src.cleaning import (
    CATEGORICAL_COLUMNS,
    LEAKAGE_COLUMN,
    NUMERIC_COLUMNS,
    RAW_DIR,
    TARGET,
    load_raw,
)

logger = logging.getLogger(__name__)

REPORT_DIR = Path("artifacts/reports")
JSON_REPORT_PATH = REPORT_DIR / "data_quality_report.json"
MD_REPORT_PATH = REPORT_DIR / "data_quality_report.md"


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a data-quality profile for ``df``."""
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    unknown_counts = {
        col: int((df[col] == "unknown").sum())
        for col in categorical_cols
        if is_string_dtype(df[col])
    }

    target_counts = df[TARGET].value_counts().to_dict() if TARGET in df.columns else {}
    yes_count = int(target_counts.get("yes", 0))
    no_count = int(target_counts.get("no", 0))

    outlier_info: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        outlier_info[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "n_outliers": n_outliers,
            "outlier_share": float(n_outliers / len(series)),
        }

    report: dict[str, Any] = {
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
        },
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "unknown_counts": unknown_counts,
        "total_unknown": int(sum(unknown_counts.values())),
        "duplicate_rows": int(df.duplicated().sum()),
        "cardinality": {
            col: int(df[col].nunique(dropna=True)) for col in df.columns
        },
        "class_imbalance": {
            "yes": yes_count,
            "no": no_count,
            "yes_share": float(yes_count / len(df)) if len(df) else 0.0,
            "no_share": float(no_count / len(df)) if len(df) else 0.0,
            "imbalance_ratio_no_to_yes": float(no_count / yes_count) if yes_count else None,
        },
        "outliers": outlier_info,
        "leakage_column": LEAKAGE_COLUMN,
        "leakage_note": (
            f"{LEAKAGE_COLUMN} is excluded from predictive features because call "
            "duration is unknown before a call is performed and directly leaks the "
            "outcome (Moro et al., 2014)."
        ),
    }
    return report


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [fmt(headers), "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def write_report(report: dict[str, Any], df: pd.DataFrame) -> None:
    """Write the data-quality report as JSON and Markdown."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote JSON report to %s", JSON_REPORT_PATH)

    lines: list[str] = []
    lines.append("# Data Quality Report — Bank Marketing (bank-additional-full.csv)")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append(f"- Rows: {report['dataset']['n_rows']:,}")
    lines.append(f"- Columns: {report['dataset']['n_columns']}")
    lines.append(f"- Target column: `{TARGET}`")
    lines.append("")
    lines.append("## Class imbalance")
    ci = report["class_imbalance"]
    lines.append(
        f"- yes (subscribed): {ci['yes']:,} ({ci['yes_share']:.2%})"
    )
    lines.append(f"- no (not subscribed): {ci['no']:,} ({ci['no_share']:.2%})")
    if ci["imbalance_ratio_no_to_yes"]:
        lines.append(f"- Imbalance ratio (no : yes): 1 : {ci['imbalance_ratio_no_to_yes']:.1f}")
    lines.append("")
    lines.append("## Duplicates")
    lines.append(f"- Exact duplicate rows: {report['duplicate_rows']:,}")
    lines.append("")
    lines.append('## ``"unknown"`` values per categorical column')
    lines.append("")
    unknown_rows = [
        (col, f"{report['unknown_counts'].get(col, 0):,}")
        for col in CATEGORICAL_COLUMNS
        if col in df.columns
    ]
    lines.append(_markdown_table(["Column", '"unknown" count'], unknown_rows))
    lines.append('- Total ``"unknown"`` cells: ' f"{report['total_unknown']:,}")
    lines.append("")
    lines.append("## Cardinality (unique values per column)")
    card_rows = [
        (col, str(report["cardinality"][col]))
        for col in df.columns
        if col in report["cardinality"]
    ]
    lines.append(_markdown_table(["Column", "Cardinality"], card_rows))
    lines.append("")
    lines.append("## Outliers (IQR method, 1.5 * IQR fences)")
    out_rows = []
    for col, info in report["outliers"].items():
        out_rows.append(
            [
                col,
                f"{info['q1']:.2f}",
                f"{info['q3']:.2f}",
                f"{info['lower_fence']:.2f}",
                f"{info['upper_fence']:.2f}",
                f"{info['n_outliers']:,}",
                f"{info['outlier_share']:.2%}",
            ]
        )
    lines.append(
        _markdown_table(
            ["Column", "Q1", "Q3", "Lower fence", "Upper fence", "Outliers", "Share"],
            out_rows,
        )
    )
    lines.append("")
    lines.append("## Cleaning and modelling decisions")
    lines.append(f"- `{LEAKAGE_COLUMN}` dropped: call duration is unknown before a call "
                 "and leaks the outcome (Moro et al., 2014).")
    lines.append('- ``"unknown"`` categorical values are treated as explicit missingness '
                 '(NaN); modelling encodes them as a separate ``"missing"`` category.')
    lines.append("- `pdays == 999` (never previously contacted) is encoded via a binary "
                 "`was_previously_contacted` indicator plus median imputation; it is "
                 "not used as a literal day count.")
    lines.append("")
    lines.append("## dtypes")
    lines.append("")
    lines.append(_markdown_table(
        ["Column", "dtype"],
        [[col, str(dtype)] for col, dtype in report["dtypes"].items()],
    ))

    MD_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown report to %s", MD_REPORT_PATH)


def run(path: Path | None = None) -> dict[str, Any]:
    """End-to-end profiling entry point: load raw, profile, write reports."""
    df = load_raw(path)
    report = profile_dataframe(df)
    write_report(report, df)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    )
    run()
