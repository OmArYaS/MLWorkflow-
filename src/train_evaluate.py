"""Baseline modelling and evaluation for the Bank Marketing dataset.

Trains a Logistic Regression baseline plus a Random Forest comparison model
using a stratified train/test split. Preprocessing (imputation, scaling,
encoding) is fitted strictly on training data via a scikit-learn
``Pipeline``, so cross-validation and holdout evaluation are leakage-free.

Class imbalance is handled with ``class_weight="balanced"`` and evaluated with
metrics appropriate for imbalanced binary classification: Precision, Recall,
F1 and ROC-AUC in addition to Accuracy.

Saves the best pipeline (by holdout ROC-AUC) to ``artifacts/models/best_pipeline.pkl``
and exports an evaluation report to ``artifacts/reports/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.cleaning import PROCESSED_PATH, run as run_cleaning
from src.features import (
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    get_feature_names,
    split_xy,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/models")
REPORT_DIR = Path("artifacts/reports")
MODEL_PATH = MODEL_DIR / "best_pipeline.pkl"
METRICS_PATH = REPORT_DIR / "evaluation_metrics.json"
REPORT_PATH = REPORT_DIR / "evaluation_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

MODELS = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=None,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def holdout_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    """Compute standard binary classification metrics for the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_model(
    name: str,
    classifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Cross-validate on train, fit on train, evaluate on holdout."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise",
    )

    cv_summary = {
        metric: {
            "mean": float(cv_results[f"test_{metric}"].mean()),
            "std": float(cv_results[f"test_{metric}"].std()),
        }
        for metric in SCORING
    }

    # Fit on the full training split (preprocessor fitted inside the pipeline,
    # strictly on training data), then evaluate on the untouched holdout set.
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    holdout = holdout_metrics(y_test, y_pred, y_prob)

    result: dict[str, Any] = {
        "model": name,
        "cv": cv_summary,
        "holdout": holdout,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "support": {"train": int(len(y_train)), "test": int(len(y_test))},
    }
    logger.info(
        "%s | CV ROC-AUC: %.4f | Holdout ROC-AUC: %.4f, F1: %.4f",
        name,
        cv_summary["roc_auc"]["mean"],
        holdout["roc_auc"],
        holdout["f1"],
    )
    return result, pipeline


def _best_pipeline_artifacts(pipeline: Pipeline, X_test: pd.DataFrame) -> dict[str, Any]:
    """Extract top features / coefficients from the fitted best pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    artifact: dict[str, Any] = {"n_features": len(feature_names)}
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        order = importances.argsort()[::-1]
        artifact["kind"] = "feature_importances"
        artifact["top_features"] = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in order[:20]
        ]
    elif hasattr(classifier, "coef_"):
        coefs = classifier.coef_.ravel()
        order = abs(coefs).argsort()[::-1]
        artifact["kind"] = "coefficients"
        artifact["top_features"] = [
            {"feature": feature_names[i], "coefficient": float(coefs[i])}
            for i in order[:20]
        ]
    return artifact


def write_markdown_report(results: dict[str, Any], best_name: str) -> None:
    """Write a human-readable evaluation report."""
    lines: list[str] = []
    lines.append("# Model Evaluation Report — Bank Marketing")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Target: ``y`` (subscribed to a term deposit: yes/no)")
    lines.append("- Split: stratified train/test (80/20), random_state=42")
    lines.append("- Cross-validation: stratified 5-fold on training data only")
    lines.append("- Class imbalance handling: ``class_weight='balanced'``")
    lines.append("- Leakage control: ``duration`` dropped; preprocessing fitted on train only")
    lines.append("")
    lines.append("## Class balance (full data)")
    lines.append(f"- Train size: {results['train_support']:,}")
    lines.append(f"- Test size: {results['test_support']:,}")
    lines.append("")
    lines.append("## Models compared")
    lines.append("")
    lines.append("| Model | CV ROC-AUC (mean ± std) | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC |")
    lines.append("|-------|-------------------------|------------------|-------------------|----------------|------------|-----------------|")
    for name in results["models"]:
        m = results["models"][name]
        cv = m["cv"]["roc_auc"]
        h = m["holdout"]
        lines.append(
            f"| {name} | {cv['mean']:.4f} ± {cv['std']:.4f} | "
            f"{h['accuracy']:.4f} | {h['precision']:.4f} | {h['recall']:.4f} | "
            f"{h['f1']:.4f} | {h['roc_auc']:.4f} |"
        )
    lines.append("")
    lines.append(f"## Best model: {best_name}")
    best = results["models"][best_name]
    h = best["holdout"]
    lines.append("")
    lines.append(f"- Selected by highest holdout ROC-AUC ({h['roc_auc']:.4f}).")
    lines.append(f"- Holdout confusion matrix (rows: true, cols: predicted): {best['confusion_matrix']}")
    lines.append("")
    lines.append("## Cross-validation detail (training data only)")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|-------|----------|-----------|--------|----|---------|")
    for name in results["models"]:
        cv = results["models"][name]["cv"]
        lines.append(
            f"| {name} | {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f} | "
            f"{cv['precision']['mean']:.4f} ± {cv['precision']['std']:.4f} | "
            f"{cv['recall']['mean']:.4f} ± {cv['recall']['std']:.4f} | "
            f"{cv['f1']['mean']:.4f} ± {cv['f1']['std']:.4f} | "
            f"{cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- ``duration`` was excluded from all features to avoid post-call data leakage.")
    lines.append('- ``"unknown"`` categorical values are modelled as an explicit '
                 '"missing" category.')
    lines.append("- ``pdays == 999`` is encoded via a ``was_previously_contacted`` indicator plus median imputation.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", REPORT_PATH)


def run(processed_path: Path = PROCESSED_PATH) -> dict[str, Any]:
    """Run the full training and evaluation workflow."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists():
        logger.warning("Processed data not found; running cleaning first")
        run_cleaning()

    df = pd.read_csv(processed_path)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Stratified split: train=%d, test=%d (positive share train=%.3f, test=%.3f)",
        len(y_train),
        len(y_test),
        float(y_train.mean()),
        float(y_test.mean()),
    )

    results: dict[str, Any] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, classifier in MODELS.items():
        result, pipeline = evaluate_model(
            name, classifier, X_train, y_train, X_test, y_test
        )
        results[name] = result
        pipelines[name] = pipeline

    # Select best model by holdout ROC-AUC.
    best_name = max(results, key=lambda n: results[n]["holdout"]["roc_auc"])
    best_pipeline = pipelines[best_name]
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Saved best pipeline (%s) to %s", best_name, MODEL_PATH)

    report = {
        "best_model": best_name,
        "train_support": int(len(y_train)),
        "test_support": int(len(y_test)),
        "positive_share_train": float(y_train.mean()),
        "positive_share_test": float(y_test.mean()),
        "models": results,
        "best_model_features": _best_pipeline_artifacts(best_pipeline, X_test),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote metrics to %s", METRICS_PATH)

    write_markdown_report(report, best_name)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    )
    run()
