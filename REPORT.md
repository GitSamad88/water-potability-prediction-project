# Water Potability — Analysis Report

## 1. Objective

Predict whether a water sample is **potable** (safe to drink) from nine physicochemical measurements,
using the [Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
(Kaggle, 3,276 samples). This report summarizes the methodology and findings of `water_potability.ipynb`.

## 2. Dataset

| | |
|---|---|
| Samples | 3,276 |
| Features | 9 (all continuous) |
| Target | `Potability` — binary, 1 = potable, 0 = not potable |
| Class balance | ~39% potable / ~61% not potable |

**Features:** `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`,
`Trihalomethanes`, `Turbidity`.

**Missing values:**

| Column | Missing | % |
|---|---|---|
| `ph` | 491 | 15.0% |
| `Sulfate` | 781 | 23.8% |
| `Trihalomethanes` | 162 | 4.9% |

No duplicate rows were found.

## 3. Methodology

1. **Cleaning** — rows with any missing value were dropped, reducing the dataset from 3,276 to **2,011**
   complete rows (a loss of ~38.6% of the data). This is the simplest possible approach; see
   [Limitations](#6-limitations--recommendations) for alternatives.
2. **EDA** — an automated profiling report was generated with `ydata-profiling`
   (`water_potability_data_profile.html`) covering distributions, correlations, and data-quality warnings
   for every column.
3. **Train/test split** — 80/20, `random_state=42`, giving 1,608 training and 403 test rows.
4. **Baseline models** — Logistic Regression, Random Forest, Decision Tree, XGBoost, and SVM, each
   trained twice: once on raw features and once on features standardized with `StandardScaler`.
5. **Hyperparameter tuning** — `GridSearchCV` (5-fold CV) for Logistic Regression (`C`), Random Forest
   (`n_estimators`, `max_depth`), and XGBoost (`max_depth`, `learning_rate`, `n_estimators`, `subsample`,
   scored directly on ROC AUC).
6. **Evaluation metrics** — Accuracy, F1 (positive/potable class), ROC AUC, and Average Precision. Given
   the class imbalance, ROC AUC and F1 are more informative than accuracy alone.

## 4. Results

### 4.1 Raw vs. scaled features

| Model | Features | Accuracy | F1 | ROC AUC | Avg. Precision |
|---|---|---|---|---|---|
| Logistic Regression | raw | 0.576 | 0.012 | 0.489 | 0.480 |
| Random Forest | raw | 0.665 | 0.506 | **0.701** | 0.645 |
| Decision Tree | raw | 0.635 | **0.548** | 0.620 | 0.507 |
| XGBoost | raw | 0.638 | 0.529 | 0.657 | 0.627 |
| SVM | raw | 0.573 | 0.000 | 0.499 | 0.440 |
| Logistic Regression | scaled | 0.571 | 0.000 | 0.465 | 0.427 |
| Random Forest | scaled | 0.655 | 0.509 | 0.692 | 0.661 |
| Decision Tree | scaled | 0.628 | 0.546 | 0.614 | 0.502 |
| XGBoost | scaled | 0.638 | 0.529 | 0.657 | 0.627 |
| **SVM** | **scaled** | **0.673** | 0.492 | **0.725** | 0.656 |

**Key observation:** scaling has almost no effect on the tree-based models (Random Forest, Decision Tree,
XGBoost), as expected since they split on raw thresholds rather than distances or gradients. It makes a
**large** difference for SVM (ROC AUC 0.499 → 0.725) and a smaller one for Logistic Regression. Without
scaling, SVM and Logistic Regression both collapse to predicting the majority class for nearly every
sample (F1 ≈ 0), illustrating how scale-sensitive these algorithms are on data where features span very
different ranges (e.g. `Solids` ~0–60,000 vs. `Turbidity` ~1–7).

### 4.2 Hyperparameter tuning

| Model | Tuning | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | `GridSearchCV` over `C` | 0.571 | 0.000 | 0.465 |
| Random Forest | `GridSearchCV` over `n_estimators`, `max_depth` | 0.648 | 0.388 | 0.693 |
| XGBoost | `GridSearchCV` over 4 params, scored on ROC AUC | 0.663 | – | **0.685** |

- **Logistic Regression** tuning did not help — varying the regularization strength `C` doesn't change
  the fundamental issue that the classes are not linearly separable in this feature space.
- **Random Forest** tuning *improved* ROC AUC slightly (0.692 → 0.693) but *hurt* F1 substantially
  (0.509 → 0.388). The grid caps `max_depth` at 7, and since the default `GridSearchCV` scoring is
  accuracy, it favors a model with high precision/low recall on the minority class over one with
  balanced precision/recall — a good example of why the CV scoring metric should match the metric you
  actually care about.
- **XGBoost** tuning (scored directly on ROC AUC) improved both accuracy (0.638 → 0.663) and ROC AUC
  (0.657 → 0.685) over the untuned baseline.

### 4.3 Best model

Across every experiment in this notebook, the **highest ROC AUC** (0.725) came from an **SVM trained on
scaled features** with default hyperparameters (it was not included in the grid search). The
**ROC-AUC-tuned XGBoost** is a close second (0.685) and is arguably a safer choice in practice since it
was selected by cross-validation rather than a single train/test split. The **Decision Tree** has the best
raw F1 score (0.548) but trails on accuracy and AUC.

**No model here is strong enough for production use as-is** — at best, ~67–70% accuracy and a ROC AUC
around 0.7, on a binary task with notable class imbalance.

## 5. Notable Issues Found in the Original Notebook

The version of this notebook originally provided had several bugs that prevented it from running and a
few correctness issues; all are fixed in the version included in this repository. See `README.md` for the
itemized list. The most important:

- **`SyntaxError` on import** — trailing commas left `StandardScaler,` and `Kfold,` (also misspelled —
  should be `KFold`) as incomplete import statements, which would crash the notebook on the very first
  cell.
- **Stale-variable bug** — a later cell printed `f"Evaluating {name}..."` relying on a leftover loop
  variable `name` from an earlier, unrelated `for` loop, rather than stating the model name directly. It
  happened to print the right label only because of the order cells were executed in — not because the
  code was actually correct.
- **Unused imports** (`LinearRegression`, `os`, `sys`, `re`) — harmless, but worth removing for clarity.

## 6. Limitations & Recommendations

- **Data loss from `dropna()`** — dropping all rows with any missing value discards ~38.6% of the
  dataset. Consider median/KNN/iterative imputation instead, especially for `Sulfate` (24% missing),
  to retain more training data.
- **Class imbalance (61/39)** is not addressed. Try `class_weight="balanced"` (Logistic Regression,
  SVM, Random Forest, Decision Tree), `scale_pos_weight` (XGBoost), or resampling (SMOTE / undersampling).
- **No probability-threshold tuning** — all models use the default 0.5 cutoff. Given the cost of a false
  negative (calling contaminated water "potable") likely outweighs a false positive, tuning the decision
  threshold using the precision-recall curve (already imported as `precision_recall_curve` but unused)
  could materially improve real-world usefulness.
- **No reproducibility for tree/kernel models** — `RandomForestClassifier()`, `DecisionTreeClassifier()`,
  `XGBClassifier()`, and `SVC()` are all instantiated without a `random_state`, so results will vary
  slightly between runs.
- **Limited hyperparameter search** — Decision Tree and SVM were never tuned at all in this notebook;
  both are reasonable candidates for a `GridSearchCV` / `RandomizedSearchCV` pass.
- **No feature engineering** — interaction terms, polynomial features, or domain-informed ratios
  (e.g. chloramine-to-organic-carbon) were not explored and might help linear models in particular.
- **Single train/test split for final comparison** — results would be more robust reported as
  cross-validated means ± standard deviation rather than a single 80/20 split.

## 7. Conclusion

Tree-based ensembles (Random Forest, XGBoost) and a properly-scaled SVM are the most competitive models
on this dataset, reaching ROC AUC ≈ 0.69–0.73. Feature scaling is essential for SVM and Logistic
Regression but irrelevant for tree-based models. The dataset's class imbalance and the amount of data lost
to missing-value removal are the two most promising levers for improving on these results.
