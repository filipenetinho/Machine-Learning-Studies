# 📚 Studying Data Science — FIA

Study notes for the **Advanced MBA in Analytics & AI — Data Science** at **FIA Business School (Labdata)**, rewritten in English, in my own words, from the course materials.

Each module has:
- **Concepts**, with the *why* behind each technique, not just the *how*
- **Formulas**, with an explanation of each term
- **The class case studies**, with their numbers and conclusions
- **Pseudocode** for the main algorithms, plus a short Python reference
- **Interview questions** with short answers

---

## 🗺️ Roadmap

```
            ┌───────────────────────── ARTIFICIAL INTELLIGENCE I ─────────────────────────┐
 1.1 EDA ──► 1.2 Modeling ──► 1.3 Linear ──► 1.4 Validation ──► 1.5 Regression ──► 1.6 RF &
 & DataViz    concepts         Regression     & Regularization    Trees             Boosting
                                                                                       │
                                                       1.7 Applied Python (Polars, Optuna, PSI)
                                                                                       │
            ┌───────────────────────── ARTIFICIAL INTELLIGENCE II ────────────────────────┐
 2.1 Time ──► 2.2 Forecasting ──► 2.3 Logistic ──► 2.4 Decision ──► 2.5 RF & Boosting ──► 2.6 Clustering
 series intro   (ARIMA, Prophet)   Regression       Trees            (classification)     (unsupervised)
```

---

## 📂 Contents

### 01 — Artificial Intelligence I

| # | Topic | Key ideas |
|---|---|---|
| 1.1 | [Exploratory Data Analysis and Data Visualization](01%20-%20Artificial%20Intelligence%20I/1.1%20-%20Exploratory%20Data%20Analysis%20and%20Data%20Visualization/README.md) | Variable types, grain, duplicates, missing values, summary measures, outliers, crosstabs, chart choice |
| 1.2 | [Introduction to Modeling](01%20-%20Artificial%20Intelligence%20I/1.2%20-%20Introduction%20to%20Modeling/README.md) | Population vs. sample, probability, sampling, supervised vs. unsupervised, predictive vs. diagnostic, drift |
| 1.3 | [Linear Regression](01%20-%20Artificial%20Intelligence%20I/1.3%20-%20Linear%20Regression/README.md) | OLS, inference (CI, p-value), R², MAE/MAPE/RMSE, residuals, Cook's distance, stepwise, VIF, dummies, transformations |
| 1.4 | [Validation and Regularization](01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/README.md) | Over/underfitting, bias–variance, holdout + k-fold, Ridge/LASSO/Elastic Net, SGD, hyperparameter search |
| 1.5 | [Regression Trees](01%20-%20Artificial%20Intelligence%20I/1.5%20-%20Regression%20Trees/README.md) | CART, impurity reduction, stopping criteria, overfitting, impurity/permutation importance, SHAP |
| 1.6 | [Random Forest and Boosting for Regression](01%20-%20Artificial%20Intelligence%20I/1.6%20-%20Random%20Forest%20and%20Boosting%20for%20Regression/README.md) | Bagging, AdaBoost, gradient boosting, HistGB, XGBoost, LightGBM, CatBoost |
| 1.7 | [Applied Python for Modeling](01%20-%20Artificial%20Intelligence%20I/1.7%20-%20Applied%20Python%20for%20Modeling/README.md) | Polars, Plotly, Optuna (TPE + pruning), segmented errors, prediction intervals, PSI and drift |

### 02 — Artificial Intelligence II

| # | Topic | Key ideas |
|---|---|---|
| 2.1 | [Introduction to Time Series Analysis](02%20-%20Artificial%20Intelligence%20II/2.1%20-%20Introduction%20to%20Time%20Series%20Analysis/README.md) | Trend, seasonality, decomposition, smoothing, stationarity (ADF/KPSS), ACF/PACF |
| 2.2 | [Time Series Forecasting Models](02%20-%20Artificial%20Intelligence%20II/2.2%20-%20Time%20Series%20Forecasting%20Models/README.md) | Baselines, AR/MA/ARMA, ARIMA, SARIMA, Prophet, SARIMAX, trees with lag features, TimeSeriesSplit |
| 2.3 | [Classification — Logistic Regression](02%20-%20Artificial%20Intelligence%20II/2.3%20-%20Classification%20-%20Logistic%20Regression/README.md) | Logistic function, odds and log-odds, confusion matrix, ROC/AUC, imbalance and cutoffs, regularization |
| 2.4 | [Classification — Decision Trees](02%20-%20Artificial%20Intelligence%20II/2.4%20-%20Classification%20-%20Decision%20Trees/README.md) | Gini and entropy, class weights, stratification, calibration (Brier, isotonic), multinomial trees |
| 2.5 | [Classification — Random Forest and Boosting](02%20-%20Artificial%20Intelligence%20II/2.5%20-%20Classification%20-%20Random%20Forest%20and%20Boosting/README.md) | Ensembles for classification, log-loss pseudo-residuals, softmax, sample weights |
| 2.6 | [Segmentation Models — Clustering](02%20-%20Artificial%20Intelligence%20II/2.6%20-%20Segmentation%20Models%20-%20Clustering/README.md) | Euclidean and Gower distance, z-score, hierarchical (linkages, dendrogram), k-means, DBSCAN |

### 🎯 [Interview Cheat Sheet](Interview%20Cheat%20Sheet.md)
A one-page review of the whole course: which model to use when, the key formulas, which metric to use for which problem, and the most common interview traps.

---

## 🧭 How to study with these notes

1. **Read the concept sections first.** Every technique has a 💡 note explaining the idea behind it.
2. **Trace the pseudocode by hand.** If you can run k-means or a tree split on paper, you understand it.
3. **Redo the class cases** in the notebooks, comparing your numbers with the ones in the notes.
4. **Answer the interview questions out loud** before reading the answers.
5. Finish with the **cheat sheet**.

---

## 📖 Main references

- James, G.; Witten, D.; Hastie, T.; Tibshirani, R.; Taylor, J. *An Introduction to Statistical Learning — with Applications in Python*. Springer, 2023.
- Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*. Springer, 2009.
- Hyndman, R. J.; Athanasopoulos, G. et al. *Forecasting: Principles and Practice, the Pythonic Way*. OTexts, 2024. <https://otexts.com/fpppy>
- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. LTC, 2022.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*. O'Reilly, 2019.

---

> ⚠️ **Disclaimer:** these are personal study notes, written in my own words from the FIA Business School / Labdata course. Course slides, datasets and notebooks are not reproduced here; all rights to the original materials belong to FIA Labdata.
