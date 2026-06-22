# 💧 Water Potability Prediction

Predicting whether a water sample is safe to drink (**potable**) from nine physicochemical
measurements, using classic ML classifiers (Logistic Regression, Random Forest, Decision Tree, XGBoost,
SVM).

## 📂 Repository Contents

```
.
├── water_potability.ipynb              # Main analysis notebook (commented, bug-fixed)
├── REPORT.md                           # Write-up: methodology, results, limitations
├── README.md                           # You are here
├── requirements.txt                    # Python dependencies
└── water_potability.csv                # Dataset (not included — see "Dataset" below)
```

## 📊 Dataset

This project uses the **Water Potability** dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/adityakadiwal/water-potability

| | |
|---|---|
| Rows | 3,276 |
| Features | 9 continuous physicochemical measurements |
| Target | `Potability` (1 = potable, 0 = not potable) |

Download `water_potability.csv` from the link above and place it in the project's root directory before
running the notebook.

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/water-potability-analysis.git
cd water-potability-analysis

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset (see "Dataset" above), then launch Jupyter
jupyter notebook water_potability.ipynb
```

## 🧪 What's Inside the Notebook

1. **Data loading & exploration** — shape, dtypes, summary statistics, missing-value audit
2. **Cleaning** — drops incomplete rows (3,276 → 2,011 rows)
3. **Automated EDA** — generates an interactive `ydata-profiling` HTML report
4. **Baseline modeling** — 5 classifiers, trained on both raw and standardized features
5. **Hyperparameter tuning** — `GridSearchCV` for Logistic Regression, Random Forest, and XGBoost
6. **Evaluation** — accuracy, F1, ROC AUC, average precision, and confusion matrices for every run

➡️ Full results and discussion are in **[`REPORT.md`](./REPORT.md)**.

## 📈 Headline Results

| Model | Features | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Random Forest | raw | 0.665 | 0.506 | 0.701 |
| Decision Tree | raw | 0.635 | **0.548** | 0.620 |
| XGBoost (tuned) | raw | 0.663 | – | 0.685 |
| **SVM** | **scaled** | **0.673** | 0.492 | **0.725** |

No model here reaches production-grade performance on its own — see [`REPORT.md`](./REPORT.md#6-limitations--recommendations)
for why, and for concrete next steps (handling class imbalance, smarter imputation, threshold tuning).

## 🛠️ Tech Stack

- Python 3.12
- pandas, NumPy
- scikit-learn, XGBoost
- matplotlib, seaborn
- ydata-profiling

## 🤝 Contributing

Issues and pull requests are welcome — see the "Limitations & Recommendations" section of
[`REPORT.md`](./REPORT.md) for ideas on where this analysis could be extended (resampling techniques,
threshold tuning, additional feature engineering, etc.).

## 📄 License

This project is provided under the MIT License. The dataset itself is subject to its own license/terms
on Kaggle — see the [dataset page](https://www.kaggle.com/datasets/adityakadiwal/water-potability) for
details.
