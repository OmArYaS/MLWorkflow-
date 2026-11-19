# Data Quality Report — Bank Marketing (bank-additional-full.csv)

## Dataset overview
- Rows: 41,188
- Columns: 21
- Target column: `y`

## Class imbalance
- yes (subscribed): 4,640 (11.27%)
- no (not subscribed): 36,548 (88.73%)
- Imbalance ratio (no : yes): 1 : 7.9

## Duplicates
- Exact duplicate rows: 12

## ``"unknown"`` values per categorical column

| Column      | "unknown" count |
|-------------|-----------------|
| job         | 330             |
| marital     | 80              |
| education   | 1,731           |
| default     | 8,597           |
| housing     | 990             |
| loan        | 990             |
| contact     | 0               |
| month       | 0               |
| day_of_week | 0               |
| poutcome    | 0               |
- Total ``"unknown"`` cells: 12,718

## Cardinality (unique values per column)
| Column         | Cardinality |
|----------------|-------------|
| age            | 78          |
| job            | 12          |
| marital        | 4           |
| education      | 8           |
| default        | 3           |
| housing        | 3           |
| loan           | 3           |
| contact        | 2           |
| month          | 10          |
| day_of_week    | 5           |
| duration       | 1544        |
| campaign       | 42          |
| pdays          | 27          |
| previous       | 8           |
| poutcome       | 3           |
| emp.var.rate   | 10          |
| cons.price.idx | 26          |
| cons.conf.idx  | 26          |
| euribor3m      | 316         |
| nr.employed    | 11          |
| y              | 2           |

## Outliers (IQR method, 1.5 * IQR fences)
| Column         | Q1      | Q3      | Lower fence | Upper fence | Outliers | Share  |
|----------------|---------|---------|-------------|-------------|----------|--------|
| age            | 32.00   | 47.00   | 9.50        | 69.50       | 469      | 1.14%  |
| duration       | 102.00  | 319.00  | -223.50     | 644.50      | 2,963    | 7.19%  |
| campaign       | 1.00    | 3.00    | -2.00       | 6.00        | 2,406    | 5.84%  |
| pdays          | 999.00  | 999.00  | 999.00      | 999.00      | 1,515    | 3.68%  |
| previous       | 0.00    | 0.00    | 0.00        | 0.00        | 5,625    | 13.66% |
| emp.var.rate   | -1.80   | 1.40    | -6.60       | 6.20        | 0        | 0.00%  |
| cons.price.idx | 93.08   | 93.99   | 91.70       | 95.37       | 0        | 0.00%  |
| cons.conf.idx  | -42.70  | -36.40  | -52.15      | -26.95      | 447      | 1.09%  |
| euribor3m      | 1.34    | 4.96    | -4.08       | 10.39       | 0        | 0.00%  |
| nr.employed    | 5099.10 | 5228.10 | 4905.60     | 5421.60     | 0        | 0.00%  |

## Cleaning and modelling decisions
- `duration` dropped: call duration is unknown before a call and leaks the outcome (Moro et al., 2014).
- ``"unknown"`` categorical values are treated as explicit missingness (NaN); modelling encodes them as a separate ``"missing"`` category.
- `pdays == 999` (never previously contacted) is encoded via a binary `was_previously_contacted` indicator plus median imputation; it is not used as a literal day count.

## dtypes

| Column         | dtype   |
|----------------|---------|
| age            | int64   |
| job            | str     |
| marital        | str     |
| education      | str     |
| default        | str     |
| housing        | str     |
| loan           | str     |
| contact        | str     |
| month          | str     |
| day_of_week    | str     |
| duration       | int64   |
| campaign       | int64   |
| pdays          | int64   |
| previous       | int64   |
| poutcome       | str     |
| emp.var.rate   | float64 |
| cons.price.idx | float64 |
| cons.conf.idx  | float64 |
| euribor3m      | float64 |
| nr.employed    | float64 |
| y              | str     |