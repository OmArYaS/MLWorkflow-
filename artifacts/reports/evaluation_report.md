# Model Evaluation Report — Bank Marketing

## Setup
- Target: ``y`` (subscribed to a term deposit: yes/no)
- Split: stratified train/test (80/20), random_state=42
- Cross-validation: stratified 5-fold on training data only
- Class imbalance handling: ``class_weight='balanced'``
- Leakage control: ``duration`` dropped; preprocessing fitted on train only

## Class balance (full data)
- Train size: 31,523
- Test size: 7,881

## Models compared

| Model | CV ROC-AUC (mean ± std) | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC |
|-------|-------------------------|------------------|-------------------|----------------|------------|-----------------|
| LogisticRegression | 0.7912 ± 0.0104 | 0.8258 | 0.3626 | 0.6500 | 0.4656 | 0.7960 |
| RandomForest | 0.7652 ± 0.0102 | 0.8664 | 0.4330 | 0.4674 | 0.4496 | 0.7710 |

## Best model: LogisticRegression

- Selected by highest holdout ROC-AUC (0.7960).
- Holdout confusion matrix (rows: true, cols: predicted): [[5910, 1051], [322, 598]]

## Cross-validation detail (training data only)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| LogisticRegression | 0.8181 ± 0.0063 | 0.3473 ± 0.0120 | 0.6343 ± 0.0172 | 0.4488 ± 0.0136 | 0.7912 ± 0.0104 |
| RandomForest | 0.8670 ± 0.0038 | 0.4357 ± 0.0143 | 0.4723 ± 0.0118 | 0.4532 ± 0.0131 | 0.7652 ± 0.0102 |

## Notes
- ``duration`` was excluded from all features to avoid post-call data leakage.
- ``"unknown"`` categorical values are modelled as an explicit "missing" category.
- ``pdays == 999`` is encoded via a ``was_previously_contacted`` indicator plus median imputation.