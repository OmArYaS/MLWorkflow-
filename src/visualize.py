"""Charts for the Bank Marketing project.

Generates, in ``artifacts/reports/``:

1. ``target_and_unknown_profile.png`` — target distribution and ``"unknown"``
   share per categorical column (data-quality view of the raw data).
2. ``model_comparison.png`` — holdout metric comparison across models.
3. ``feature_importance.png`` — top features/coefficients of the best model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from src.cleaning import CATEGORICAL_COLUMNS, PRIMARY_PATH, FALLBACK_PATH, TARGET, load_raw
from src.train_evaluate import METRICS_PATH, MODEL_PATH

logger = logging.getLogger(__name__)

REPORT_DIR = Path("artifacts/reports")
TARGET_UNKNOWN_CHART = REPORT_DIR / "target_and_unknown_profile.png"
MODEL_COMPARISON_CHART = REPORT_DIR / "model_comparison.png"
FEATURE_IMPORTANCE_CHART = REPORT_DIR / "feature_importance.png"

sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True


def _load_raw() -> "pd.DataFrame":  # type: ignore[name-defined]
    if PRIMARY_PATH.exists():
        path = PRIMARY_PATH
    elif FALLBACK_PATH.exists():
        path = FALLBACK_PATH
    else:
        raise FileNotFoundError("No raw dataset found for visualization")
    return load_raw(path)


def plot_target_and_unknown_profile(df: "pd.DataFrame", save_path: Path = TARGET_UNKNOWN_CHART) -> None:  # type: ignore[name-defined]
    """Target distribution plus ``"unknown"`` share per categorical column."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Target distribution
    ax = axes[0]
    counts = df[TARGET].value_counts()
    order = ["no", "yes"]
    counts = counts.reindex([o for o in order if o in counts.index])
    colors = ["#c44e52", "#55a868"]
    bars = ax.bar(
        [str(i) for i in counts.index],
        counts.values,
        color=colors[: len(counts)],
        width=0.6,
    )
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            f"{count:,}\n({count / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Target Distribution (term deposit subscription)", fontsize=12, fontweight="bold")
    ax.set_xlabel("y")
    ax.set_ylabel("Number of clients")
    ax.set_ylim(0, total * 1.16)

    # (b) "unknown" share per categorical column
    ax = axes[1]
    shares = []
    labels = []
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            share = float((df[col] == "unknown").mean())
            shares.append(share)
            labels.append(col)
    if shares:
        order_idx = sorted(range(len(shares)), key=lambda i: shares[i], reverse=True)
        shares = [shares[i] for i in order_idx]
        labels = [labels[i] for i in order_idx]
        bars = ax.barh(labels, shares, color="#8172b3", height=0.6)
        ax.set_xlim(0, max(shares) * 1.25 if shares else 1.0)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        for bar, share in zip(bars, shares):
            ax.annotate(
                f"{share:.1%}",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
                color="black",
                xytext=(4, 0),
                textcoords="offset points",
            )
        ax.set_title('Share of ``"unknown"`` Values per Categorical Column', fontsize=12, fontweight="bold")
        ax.set_xlabel("Share of rows")
    else:
        ax.set_title('No ``"unknown"`` values found', fontsize=12)
        ax.axis("off")

    fig.suptitle("Data Quality: Target Distribution & Missingness Profile", fontsize=14, fontweight="bold", y=1.02)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_model_comparison(
    metrics_report: dict, save_path: Path = MODEL_COMPARISON_CHART
) -> None:
    """Grouped bar chart of holdout metrics per model."""
    model_names = list(metrics_report["models"].keys())
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = {
        key: [metrics_report["models"][name]["holdout"][key] for name in model_names]
        for key in metric_keys
    }

    x = range(len(model_names))
    width = 0.15
    offsets = [-(2 * width), -width, 0, width, 2 * width]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", n_colors=len(metric_keys))
    for idx, key in enumerate(metric_keys):
        ax.bar(
            [pos + offsets[idx] for pos in x],
            values[key],
            width,
            label=key.replace("_", " ").title(),
            color=colors[idx],
        )
        for pos, val in zip(x, values[key]):
            ax.annotate(
                f"{val:.3f}",
                (pos + offsets[idx], val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_ylabel("Score")
    ax.set_title("Holdout Metric Comparison by Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_feature_importance(
    pipeline, save_path: Path = FEATURE_IMPORTANCE_CHART, top_n: int = 15
) -> None:
    """Top feature importances (Random Forest) or coefficients (Logistic Regression)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    from src.features import get_feature_names

    feature_names = get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        kind = "importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_.ravel()
        kind = "coefficient"
    else:
        logger.warning("Classifier exposes neither feature_importances_ nor coef_; skipping chart")
        return

    importance_df = pd.DataFrame({"feature": feature_names, "value": values})
    importance_df["abs"] = importance_df["value"].abs()
    top = importance_df.nlargest(top_n, "abs").sort_values("abs", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bar_colors = ["#55a868" if v >= 0 else "#c44e52" for v in top["value"]]
    ax.barh(top["feature"], top["value"], color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(kind.title())
    ax.set_title(f"Top {top_n} Features by {kind.title()} — {classifier.__class__.__name__}", fontsize=13, fontweight="bold")
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['value']:+.3f}",
            (row["value"], row.name),
            ha="left" if row["value"] >= 0 else "right",
            va="center",
            fontsize=9,
            xytext=(4 if row["value"] >= 0 else -4, 0),
            textcoords="offset points",
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def run() -> None:
    """Generate all charts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_raw()
    plot_target_and_unknown_profile(df)

    if METRICS_PATH.exists():
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        plot_model_comparison(report)
    else:
        logger.warning("Skipping model comparison chart: %s not found", METRICS_PATH)

    if MODEL_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)
        plot_feature_importance(pipeline)
    else:
        logger.warning("Skipping feature importance chart: %s not found", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    )
    run()
