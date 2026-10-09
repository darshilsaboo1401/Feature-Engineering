
# Feature Engineering

A beginner-friendly repository to learn **Feature Engineering** for Machine Learning — starting from the basics and building up with hands-on Jupyter notebooks.

---

## Table of Contents

1. [What is Feature Engineering?](#1-what-is-feature-engineering)
2. [Why is it Important?](#2-why-is-it-important)
3. [Main Steps in Feature Engineering](#3-main-steps-in-feature-engineering)
4. [Feature Scaling](#4-feature-scaling)
5. [Standardization (Z-score Scaling)](#5-standardization-z-score-scaling)
6. [Repository Contents](#6-repository-contents)
7. [Getting Started](#7-getting-started)
8. [Roadmap](#8-roadmap)

---

## 1. What is Feature Engineering?

**Feature Engineering** is the process of using domain knowledge to **select, create, transform, and prepare input variables (features)** so that a machine learning model can learn from the data more effectively.

- A **feature** is an individual measurable property of the data (a column in your dataset), e.g. `age`, `salary`, `city`.
- Raw data is rarely ready for a model. It may have missing values, text categories, different units, outliers, or hidden patterns that are not obvious.
- Feature engineering turns this raw data into a form that models can understand and learn from.

> "Better features beat better algorithms." A simple model with great features often outperforms a complex model with poor ones.

**Example**

| Raw feature | Engineered feature |
|---|---|
| `date_of_birth = 1998-05-12` | `age = 28` |
| `timestamp = 2026-10-09 18:30` | `hour = 18`, `is_weekend = 0` |
| `city = "Jaipur"` | One-hot encoded columns (`city_Jaipur`, `city_Delhi`, ...) |
| `height`, `weight` | `BMI = weight / height²` |

---

## 2. Why is it Important?

- **Improves model accuracy** by exposing useful patterns.
- **Handles messy data** such as missing values and outliers.
- **Makes algorithms work properly** — many models (KNN, SVM, Linear/Logistic Regression, Neural Networks, PCA) are sensitive to feature scale.
- **Reduces overfitting** by removing irrelevant or redundant features.
- **Speeds up training** through simpler, smaller inputs.
- **Improves interpretability** when features are meaningful.

---

## 3. Main Steps in Feature Engineering

| Step | What it means | Common techniques |
|---|---|---|
| **Handling missing values** | Fill or remove empty entries | Mean/median/mode imputation, KNN imputer, drop rows/columns |
| **Handling outliers** | Treat extreme values | IQR method, Z-score, capping/winsorization |
| **Encoding categorical variables** | Convert text categories to numbers | Label encoding, One-hot encoding, Ordinal encoding, Target encoding |
| **Feature scaling** | Bring features to a similar range | Standardization, Min-Max scaling, Robust scaling |
| **Feature transformation** | Change the distribution of a feature | Log, square root, Box-Cox, Yeo-Johnson, power transforms |
| **Feature creation** | Build new features from existing ones | Ratios, interactions, polynomial features, date/time parts, binning |
| **Feature selection** | Keep only the useful features | Correlation, variance threshold, chi-square, RFE, feature importance |
| **Dimensionality reduction** | Compress many features into fewer | PCA, LDA |

---

## 4. Feature Scaling

**Feature scaling** puts numerical features on a comparable scale.

Imagine a dataset with `age` (18–60) and `salary` (20,000–2,000,000). Distance-based and gradient-based algorithms will treat `salary` as far more important simply because its numbers are bigger. Scaling fixes this.

**Models that need scaling:** KNN, SVM, K-Means, Logistic/Linear Regression (with regularization or gradient descent), Neural Networks, PCA.

**Models that usually don't:** Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM).

Common scalers:

| Scaler | Formula | Range |
|---|---|---|
| **Standardization** (`StandardScaler`) | `(x − mean) / std` | Mean = 0, Std = 1 |
| **Min-Max Scaling** (`MinMaxScaler`) | `(x − min) / (max − min)` | 0 to 1 |
| **Robust Scaling** (`RobustScaler`) | `(x − median) / IQR` | Not fixed; robust to outliers |
| **MaxAbs Scaling** (`MaxAbsScaler`) | `x / max(|x|)` | −1 to 1 |

---

## 5. Standardization (Z-score Scaling)

Standardization rescales a feature so that it has a **mean of 0** and a **standard deviation of 1**.

```
z = (x − μ) / σ
```

where `μ` is the mean and `σ` is the standard deviation of the feature.

**Key points**

- It does **not** bound values to a fixed range.
- It keeps the shape of the original distribution (it does not make data normal).
- It works well when data is roughly Gaussian and is the default choice for many ML algorithms.
- It is affected by outliers (since mean and std are) — use `RobustScaler` if outliers are a concern.
- **Fit the scaler on the training set only**, then apply it to the test set to avoid data leakage.

**Quick example**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df[["age", "salary"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn mean/std from train only
X_test_scaled = scaler.transform(X_test)        # apply the same mean/std to test
```

---

## 6. Repository Contents

| File | Description |
|---|---|
| [`Scaler_Standardization.ipynb`](./Scaler_Standardization.ipynb) | Jupyter notebook demonstrating feature scaling using Standardization |
| `README.md` | Overview of feature engineering concepts (this file) |

More notebooks will be added as the repository grows.

---

## 7. Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/darshilsaboo1401/Feature-Engineering.git
cd Feature-Engineering
```

**2. Install the requirements**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Launch Jupyter**

```bash
jupyter notebook
```

Then open `Scaler_Standardization.ipynb` and run the cells in order.

---

## 8. Roadmap

- [x] Standardization
- [ ] Min-Max Scaling / Normalization
- [ ] Robust Scaling
- [ ] Handling missing values
- [ ] Encoding categorical variables
- [ ] Outlier detection and treatment
- [ ] Feature transformation (log, power transforms)
- [ ] Feature creation and binning
- [ ] Feature selection
- [ ] Dimensionality reduction (PCA)

---

## Author

**Darshil Saboo** — [GitHub](https://github.com/darshilsaboo1401)

If you find this repository helpful, consider giving it a ⭐
