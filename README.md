# Bank Marketing — Data Cleaning, Feature Engineering & Predictive Analytics

End-to-end machine learning project on the [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) dataset (`bank-additional-full.csv`, 41,188 records of Portuguese bank telemarketing campaigns, May 2008 – Nov 2010). The goal is binary classification: predict whether a client subscribes to a term deposit (`y = yes/no`).

Built as a portfolio-ready, modular Python project: profiling → deterministic cleaning → leakage-safe feature engineering → baseline + comparison modeling → evaluation → visualization, with tests.

## Citation

This dataset is publicly available for research. As requested in the dataset documentation, please cite:

> S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems (2014), doi:10.1016/j.dss.2014.03.001 ([Moro et al., 2014]).

## Project structure

```
data/raw/               # bank-additional-full.csv (or dataset.csv fallback)
data/processed/         # bank_clean.csv  (deterministically cleaned frame)
artifacts/reports/      # data_quality_report.{json,md}, evaluation_report.md,
                        # evaluation_metrics.json, charts (*.png)
artifacts/models/       # best_pipeline.pkl (fitted sklearn Pipeline, inference-ready)
src/
  profiling.py          # data profiling + data-quality report (JSON/Markdown)
  cleaning.py           # deterministic cleaning & transformation pipeline
  features.py           # sklearn ColumnTransformer/Pipeline preprocessing (train-fit only)
  train_evaluate.py     # baseline + comparison models, CV, holdout, best-model export
  visualize.py          # charts for reports
tests/
  test_pipeline.py      # smoke tests (8 tests)
requirements.txt
README.md
```

## Problem statement

Given a client's demographic, campaign, and socio-economic profile as recorded **before** the telemarketing call outcome is known, predict whether the client will subscribe to a term deposit (`y=yes`). This is a realistic screening model: it must not use information that is only available after the call (most importantly, call `duration`).

## Dataset & data quality findings

Source: `data/raw/dataset.csv` (a copy of `bank-additional-full.csv`, separated by `;`; delimiter is auto-detected, falling back to `,`). 41,188 rows × 21 columns. Target `y` is binary (`yes`/`no`).

| Finding | Detail |
|---|---|
| Class imbalance | `yes` = 4,640 (11.3%), `no` = 36,548 (88.7%); ratio ≈ 1 : 7.9 |
| Exact duplicates | 12 fully duplicated rows in the raw file |
| `"unknown"` values | 12,718 cells across 6 categorical columns: `default` 8,597 (20.9% of rows), `education` 1,731, `job` 330, `housing` 990, `loan` 990, `marital` 80 |
| `pdays == 999` | 39,673 of 41,176 pre-dedup rows (≈ 96%) were never previously contacted; 999 is a sentinel, not a day count |
| Outliers (IQR 1.5×) | `previous` 5,625 (13.7%), `campaign` 2,406 (5.8%), `pdays` 1,515 (3.7%), `age` 469 (1.1%), `cons.conf.idx` 447 (1.1%) |

Full data-quality report: `artifacts/reports/data_quality_report.{md,json}` (see `src/profiling.py`).

## Cleaning & engineering decisions

**`src/cleaning.py`** — deterministic pipeline (logs every decision):

1. **Delimiter auto-detection**: reads `data/raw/bank-additional-full.csv` if present, otherwise `data/raw/dataset.csv`; detects `;` vs `,` from the header sample.
2. **String standardization**: strips whitespace on text columns.
3. **`"unknown"` handling**: treated as explicit missingness (`NaN`), not dropped and not imputed with a guess. Downstream, missingness is modelled as its own `"missing"` one-hot category so the algorithm can learn from the fact that a value was not recorded.
4. **`pdays == 999`**: a binary `was_previously_contacted` indicator is created, then `999` is replaced with `NaN` (median-imputed downstream) — it is never used as a literal number of days.
5. **Duplicate removal**: exact duplicates removed **after** `duration` is dropped, because several raw records share the same client profile with different call durations; without `duration` they collapse into true duplicates that must not leak across train/test.
6. **`duration` dropped**: call duration is unknown before a call is performed and directly leaks the target (e.g., `duration = 0 ⇒ y = "no"`). It is excluded from all predictive features, as advised in the dataset documentation (Moro et al., 2014). This decision is documented in the code (`src/cleaning.py: LEAKAGE_COLUMN`) and the reports.

Cleaned frame: **39,404 rows × 21 columns** (`data/processed/bank_clean.csv`; 1,784 duplicates removed).

**`src/features.py`** — preprocessing fitted **strictly on training data**:

- Numeric (`age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `was_previously_contacted`): median imputation + `StandardScaler` (fitted on train only).
- Categorical (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`): missingness → explicit `"missing"` category → `OneHotEncoder(handle_unknown="ignore")`.
- Wrapped in a `ColumnTransformer` inside a `sklearn.pipeline.Pipeline`, so cross-validation never fits preprocessors on validation folds (no leakage).
- Socio-economic context attributes (employment variation rate, CPI, consumer confidence, EURIBOR, nr.employed) are retained as numeric features — Moro et al. (2014) show these substantially improve prediction even without `duration`.

## Modeling results

Numbers below are **real outputs from running the pipeline** (see `artifacts/reports/evaluation_report.md` and `evaluation_metrics.json`).

Setup: stratified train/test split 80/20 (train 31,523 / test 7,881; positive share ≈ 11.7% in both), stratified 5-fold CV on training data only, `class_weight="balanced"` for both models to handle the 1:7.9 imbalance.

| Model | CV ROC-AUC (mean ± std) | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7912 ± 0.0104 | 0.8258 | 0.3626 | 0.6500 | 0.4656 | **0.7960** |
| Random Forest (300 trees) | 0.7652 ± 0.0102 | 0.8664 | 0.4330 | 0.4674 | 0.4496 | 0.7710 |

Best model (by holdout ROC-AUC): **Logistic Regression**, saved as a full inference-ready pipeline to `artifacts/models/best_pipeline.pkl`.

Holdout confusion matrix (Logistic Regression, rows = true, cols = predicted):

| | pred no | pred yes |
|---|---|---|
| actual no | 5,910 | 1,051 |
| actual yes | 322 | 598 |

Top drivers of a subscription (Logistic Regression coefficients on the 63 one-hot/scaled features): `emp.var.rate` (−2.18), `month_mar` (+1.26), `cons.price.idx` (+1.02), `euribor3m` (+0.60), `poutcome_success` (+0.30), `was_previously_contacted` (+0.24), `job_missing` (−0.24). Economic distress (negative employment variation rate) and March contacts strongly associate with subscription; unsuccessful previous outcomes and June/November contacts negatively.

Charts: `artifacts/reports/target_and_unknown_profile.png`, `model_comparison.png`, `feature_importance.png`.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset at data/raw/bank-additional-full.csv
#    (the repo already contains data/raw/dataset.csv, which is used as fallback)

# 3. Run the full pipeline end to end
python -m src.profiling     # data-quality report -> artifacts/reports/
python -m src.cleaning      # cleaned data -> data/processed/bank_clean.csv
python -m src.train_evaluate# training, CV, holdout, best model -> artifacts/
python -m src.visualize     # charts -> artifacts/reports/

# 4. Run tests
pytest tests/test_pipeline.py -v   # 8 tests: loading, no duration leakage,
                                    # unknown/pdays handling, duplicates,
                                    # no NaNs after preprocessing, inference shape
```

## Limitations & next steps

- Single stratified train/test split; results would be firmer with nested CV or a temporal holdout (data is ordered by date).
- Logistic Regression is the best model here mostly on ROC-AUC; Random Forest achieves higher accuracy by predicting the majority class more often — precision/recall trade-offs should be tuned to the business cost of false calls vs. missed subscriptions.
- Threshold tuning, probability calibration, and hyperparameter search (e.g., `HistGradientBoostingClassifier`, regularization strength) are natural next steps.
- `"unknown"` is modelled as a `"missing"` category; alternative strategies (imputation by subgroup, dedicated missingness indicators per column) could be compared.
- The pipeline is serialized with `joblib` and can be loaded with `joblib.load("artifacts/models/best_pipeline.pkl")` and called via `.predict()` / `.predict_proba()` on new client records with the same schema.

## Repository contents

- `src/profiling.py`, `src/cleaning.py`, `src/features.py`, `src/train_evaluate.py`, `src/visualize.py` — typed, documented, logging-based modules
- `tests/test_pipeline.py` — smoke tests
- `requirements.txt` — minimal dependencies (pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, pytest)
- `artifacts/` — all generated reports, charts, and the best model

Dataset citation: Moro et al. (2014), doi:10.1016/j.dss.2014.03.001.
