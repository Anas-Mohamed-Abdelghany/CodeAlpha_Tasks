# 💳 Credit Scoring Model
### CodeAlpha Internship — Task 1

> A machine learning pipeline that predicts an individual's creditworthiness using past financial data, built with Random Forest and scikit-learn.

---

## 📌 Project Overview

Credit scoring is the backbone of modern lending decisions. This project builds a full end-to-end classification pipeline that:

- Ingests raw financial data
- Engineers a binary target (`Is_Bad_Credit`) from continuous credit scores
- Preprocesses mixed numeric/categorical features through a robust sklearn pipeline
- Trains and tunes a **Random Forest Classifier**
- Evaluates performance using **Precision, Recall, F1-Score, and Confusion Matrix**
- Interprets the model using **Permutation Importance**

---

## 📊 Dataset

**Source:** [Kaggle — Credit Scoring Dataset by SyncoraAI](https://www.kaggle.com/datasets/syncoraai/credit-scoring-dataset)

| Field | Description |
|---|---|
| `CUST_ID` | Unique customer identifier (dropped — not a feature) |
| `CREDIT_SCORE` | Continuous credit score (used to derive target, then dropped) |
| `DEFAULT` | Whether the customer defaulted (dropped — future leakage) |
| `Is_Bad_Credit` | ✅ **Target** — 1 if `CREDIT_SCORE < 600`, else 0 |
| Other columns | Income, debts, payment history, etc. |

---

## 🗂️ Project Structure

```
CodeAlpha_Task1/
│
├── CodeAlpha_Task1.ipynb       # Main notebook
├── README.md                   # This file
│
├── images/                     # 📁 Place your output images here
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_tuned.png
│   └── feature_importance.png
│
└── synthetic_e2d...csv         # Raw dataset (after Kaggle download)
```

---

## 🔄 Pipeline Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│   Data Audit    │  → Check size, types, duplicates, class balance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stratified Split│  → Train (70%) / Val (15%) / Test (15%)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│           ColumnTransformer              │
│  ┌──────────────┐  ┌───────────────────┐ │
│  │   Numeric    │  │   Categorical     │ │
│  │  - Imputer   │  │  - Imputer        │ │
│  │  - Scaler    │  │  - OneHotEncoder  │ │
│  └──────────────┘  └───────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  RandomForestClassifier│  class_weight='balanced'
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  Evaluation &    │  → Classification Report + Confusion Matrix
│  Interpretability│  → Permutation Importance (Top 10 features)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Hyperparameter   │  → RandomizedSearchCV (10 combos, 3-fold CV)
│     Tuning       │  → Best model selected by F1-Score
└──────────────────┘
```

---

## 🖼️ Results & Visualizations

> **How to add images:** After running the notebook, save the matplotlib figures and place them in the `images/` folder. Then they'll appear here automatically.

### Confusion Matrix — Baseline Model
![Baseline Confusion Matrix](images/confusion_matrix_baseline.png)

*Replace this placeholder by running the notebook and saving the figure with:*
```python
plt.savefig('images/confusion_matrix_baseline.png', dpi=150, bbox_inches='tight')
```

---

### Confusion Matrix — Tuned Model
![Tuned Confusion Matrix](images/confusion_matrix_tuned.png)

*Replace this placeholder by saving the tuned model's confusion matrix plot.*

---

### Top 10 Most Predictive Features (Permutation Importance)
![Feature Importance](images/feature_importance.png)

*Replace this placeholder by saving the feature importance bar chart:*
```python
feature_importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Features — Permutation Importance")
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
```

---

## 📈 Evaluation Metrics

The model is evaluated using the following metrics on the **Validation Set**:

| Metric | Description |
|---|---|
| **Precision** | Of all predicted bad-credit customers, how many actually were? |
| **Recall** | Of all actual bad-credit customers, how many did we catch? |
| **F1-Score** | Harmonic mean of Precision & Recall — the primary metric |
| **Confusion Matrix** | Visual breakdown of TP, FP, TN, FN |

> ⚠️ Since this is an imbalanced classification problem, `class_weight='balanced'` is used and F1-Score is prioritized over raw accuracy.

---

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` was used to search over the following space:

```python
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__max_features': ['sqrt', 'log2']
}
```

- **Strategy:** 10 random combinations, 3-fold cross-validation
- **Scoring:** F1-Score

---

## 🚀 How to Run

1. **Clone/download** the notebook file
2. **Set up Kaggle API** credentials to download the dataset:
   ```bash
   kaggle datasets download -d syncoraai/credit-scoring-dataset
   unzip credit-scoring-dataset.zip
   ```
3. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib numpy
   ```
4. **Run all cells** in `CodeAlpha_Task1.ipynb` top to bottom
5. **Save your output images** into the `images/` folder (see commands above)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading & manipulation |
| scikit-learn | Pipeline, preprocessing, modeling, evaluation |
| matplotlib | Visualization |
| Kaggle API | Dataset download |

---

## 👤 Author

**Your Name**
CodeAlpha Machine Learning Internship — Task 1
[LinkedIn](#) · [GitHub](#)

---

## 📄 License

This project is for educational purposes as part of the CodeAlpha internship program.