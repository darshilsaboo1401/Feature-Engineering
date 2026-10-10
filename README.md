# Feature Engineering

This repository provides a structured introduction to **Feature Engineering** for Machine Learning, beginning with core concepts and progressing to practical implementations in Jupyter notebooks.

---

## Table of Contents

1. [What is Feature Engineering?](#1-what-is-feature-engineering)
2. [Importance of Feature Engineering](#2-importance-of-feature-engineering)
3. [Major Stages of Feature Engineering](#3-major-stages-of-feature-engineering)
4. [Feature Scaling](#4-feature-scaling)
5. [Standardization](#5-standardization)
6. [Normalization](#6-normalization)
   - [6.1 Min-Max Scaling](#61-min-max-scaling)
   - [6.2 Mean Normalization](#62-mean-normalization)
   - [6.3 Absolute Maximum (MaxAbs) Scaling](#63-absolute-maximum-maxabs-scaling)
   - [6.4 Robust Scaling](#64-robust-scaling)
   - [6.5 Unit Vector Normalization](#65-unit-vector-normalization)
7. [Comparison of Scaling Techniques](#7-comparison-of-scaling-techniques)
8. [Choosing a Scaling Technique](#8-choosing-a-scaling-technique)
9. [Repository Contents](#9-repository-contents)
10. [Getting Started](#10-getting-started)
11. [Roadmap](#11-roadmap)

---

## 1. What is Feature Engineering?

**Feature Engineering** is the process of using domain knowledge to **select, create, transform, and prepare input variables (features)** so that a machine learning model can learn from the data effectively.

- A **feature** is an individual measurable property of the data, represented as a column in a dataset (for example, `age`, `salary`, or `city`).
- Raw data is seldom suitable for direct use in a model. It may contain missing values, categorical text, inconsistent units, outliers, or patterns that are not explicitly represented.
- Feature engineering converts raw data into a representation that machine learning algorithms can process and learn from.

**Examples**

| Raw Feature | Engineered Feature |
|---|---|
| `date_of_birth = 1998-05-12` | `age = 28` |
| `timestamp = 2026-10-09 18:30` | `hour = 18`, `is_weekend = 0` |
| `city = "Jaipur"` | One-hot encoded columns (`city_Jaipur`, `city_Delhi`, ...) |
| `height`, `weight` | `BMI = weight / height²` |

---

## 2. Importance of Feature Engineering

- **Model performance:** Well-constructed features expose useful patterns and often improve predictive accuracy more than changing the algorithm.
- **Data quality:** It addresses missing values, outliers, and inconsistent formats.
- **Algorithm compatibility:** Many algorithms, such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), K-Means, Principal Component Analysis (PCA), and neural networks, are sensitive to the scale of input features.
- **Generalization:** Removing irrelevant or redundant features can reduce overfitting.
- **Efficiency:** Compact, informative inputs reduce training time and computational cost.
- **Interpretability:** Meaningful features make model behavior easier to explain.

---

## 3. Major Stages of Feature Engineering

| Stage | Description | Common Techniques |
|---|---|---|
| **Handling missing values** | Fill in or remove absent entries | Mean/median/mode imputation, KNN imputation, row/column removal |
| **Handling outliers** | Detect and treat extreme values | IQR method, Z-score method, capping (winsorization) |
| **Encoding categorical variables** | Convert categories into numerical form | Label encoding, ordinal encoding, one-hot encoding, target encoding |
| **Feature scaling** | Bring numerical features to a comparable scale | Standardization, normalization |
| **Feature transformation** | Alter the distribution of a feature | Log, square root, Box-Cox, Yeo-Johnson |
| **Feature creation** | Derive new features from existing ones | Ratios, interaction terms, polynomial features, date/time components, binning |
| **Feature selection** | Retain the most relevant features | Correlation analysis, variance threshold, chi-square test, recursive feature elimination, feature importance |
| **Dimensionality reduction** | Represent data with fewer features | PCA, Linear Discriminant Analysis (LDA) |

---

## 4. Feature Scaling

**Feature scaling** is the process of transforming numerical features so that they lie on a comparable scale.

Consider a dataset with `age` (approximately 18 to 60) and `salary` (approximately 20,000 to 2,000,000). Algorithms that rely on distances or gradient-based optimization will give disproportionate influence to `salary` simply because its values are larger. Feature scaling removes this imbalance.

**Algorithms that generally require scaling**

- Distance-based: KNN, K-Means, SVM
- Gradient-based or regularized: Linear and Logistic Regression (when trained with gradient descent or regularization), neural networks
- Variance-based: PCA

**Algorithms that are generally insensitive to scaling**

- Decision Trees, Random Forests, and Gradient Boosting methods (e.g., XGBoost, LightGBM)

Feature scaling techniques are broadly divided into two categories:

```
Feature Scaling
├── Standardization
└── Normalization
    ├── Min-Max Scaling
    ├── Mean Normalization
    ├── Absolute Maximum (MaxAbs) Scaling
    ├── Robust Scaling
    └── Unit Vector Normalization
```

> **Terminology note:** The terms *standardization* and *normalization* are used inconsistently across the literature. In this repository, **standardization** refers to Z-score scaling, and **normalization** refers to the family of methods that rescale features using quantities such as the minimum, maximum, median, or vector norm. Robust scaling is placed under normalization here for convenience, although some sources group it with standardization because it also centers and scales data.

---

## 5. Standardization

Standardization (Z-score scaling) rescales a feature so that it has a **mean of 0** and a **standard deviation of 1**.

```
z = (x − μ) / σ
```

where `μ` is the mean and `σ` is the standard deviation of the feature.

**Characteristics**

- The output is not bounded to a fixed range.
- The shape of the original distribution is preserved; standardization does not make data normally distributed.
- It is well suited to features that are approximately Gaussian and is a common default for many algorithms.
- It is sensitive to outliers, since both the mean and the standard deviation are affected by extreme values.
- Scikit-learn's `StandardScaler` computes the population standard deviation (`ddof = 0`).

**Implementation**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn mean and std from training data only
X_test_scaled = scaler.transform(X_test)        # apply the same parameters to test data
```

> **Avoiding data leakage:** Always fit a scaler on the training set only, then use it to transform both the training and test sets.

---

## 6. Normalization

Normalization rescales feature values using reference statistics such as the minimum, maximum, median, or vector norm. The most common variants are described below.

### 6.1 Min-Max Scaling

Rescales each feature to a fixed range, typically **[0, 1]**.

```
x' = (x − x_min) / (x_max − x_min)
```

To scale to a custom range `[a, b]`:

```
x' = a + (x − x_min)(b − a) / (x_max − x_min)
```

**Characteristics**

- Preserves the shape of the original distribution.
- Guarantees a bounded output range on the training data.
- Highly sensitive to outliers, because the minimum and maximum define the scale.
- Commonly used for neural networks and image data, and when bounded inputs are required.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 6.2 Mean Normalization

Centers the data around zero and scales it by the range of the feature.

```
x' = (x − μ) / (x_max − x_min)
```

**Characteristics**

- The resulting values have a mean of 0 and lie approximately in the range [−1, 1].
- Sensitive to outliers because it uses the mean, minimum, and maximum.
- Scikit-learn does not provide a dedicated transformer for this method; it is typically implemented manually.

```python
X_scaled = (X - X.mean()) / (X.max() - X.min())
```

### 6.3 Absolute Maximum (MaxAbs) Scaling

Divides each feature by its maximum absolute value.

```
x' = x / max(abs(x))
```

**Characteristics**

- Output values lie in the range **[−1, 1]**.
- Does not shift or center the data, so **sparsity is preserved**; this makes it suitable for sparse datasets.
- Sensitive to outliers because it depends on the maximum absolute value.

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 6.4 Robust Scaling

Centers each feature on its **median** and scales it by the **interquartile range (IQR)**.

```
x' = (x − median) / IQR,    where IQR = Q3 − Q1
```

**Characteristics**

- Uses statistics that are resistant to extreme values, making it the preferred choice for data containing outliers.
- The output range is not fixed.
- By default, scikit-learn uses the 25th and 75th percentiles to compute the IQR.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 6.5 Unit Vector Normalization

Unlike the previous methods, which operate on **columns (features)**, unit vector normalization operates on each **row (sample)**, scaling it to have a norm of 1.

```
x' = x / ||x||
```

Common norms include L1 (sum of absolute values), L2 (Euclidean length), and max.

**Characteristics**

- Preserves the direction of each sample while discarding its magnitude.
- Frequently used in text classification, clustering, and similarity-based methods such as cosine similarity.

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm="l2")
X_scaled = normalizer.fit_transform(X)
```

---

## 7. Comparison of Scaling Techniques

| Technique | Formula | Output Range | Centers Data | Sensitive to Outliers | Scikit-learn Class |
|---|---|---|---|---|---|
| Standardization | (x − μ) / σ | Unbounded (mean 0, std 1) | Yes | Yes | `StandardScaler` |
| Min-Max Scaling | (x − min) / (max − min) | [0, 1] by default | No | Yes (high) | `MinMaxScaler` |
| Mean Normalization | (x − μ) / (max − min) | Approximately [−1, 1] | Yes | Yes | None (manual) |
| MaxAbs Scaling | x / max(abs(x)) | [−1, 1] | No | Yes | `MaxAbsScaler` |
| Robust Scaling | (x − median) / IQR | Unbounded | Yes | No (robust) | `RobustScaler` |
| Unit Vector Normalization | x / norm(x) | Norm of each row = 1 | No | Not applicable (row-wise) | `Normalizer` |

---

## 8. Choosing a Scaling Technique

| Situation | Recommended Technique |
|---|---|
| Features are roughly Gaussian; algorithm assumes centered data (e.g., PCA, SVM, linear models) | Standardization |
| A bounded range is required (e.g., neural networks, image pixel values) | Min-Max Scaling |
| The dataset contains significant outliers | Robust Scaling |
| The data is sparse and zeros must be preserved | MaxAbs Scaling |
| Samples are compared by direction or angle (e.g., text vectors, cosine similarity) | Unit Vector Normalization |
| Tree-based models | Scaling is typically unnecessary |

---

## 9. Repository Contents

| File | Description |
|---|---|
| [`Scaler_Standardization.ipynb`](./Scaler_Standardization.ipynb) | Jupyter notebook demonstrating feature scaling using standardization |
| `README.md` | Documentation of feature engineering concepts (this file) |

Additional notebooks will be added as the repository develops.

---

## 10. Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/darshilsaboo1401/Feature-Engineering.git
cd Feature-Engineering
```

**2. Install the dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open `Scaler_Standardization.ipynb` and run the cells in order.

---

## 11. Roadmap

- [x] Standardization
- [ ] Normalization: Min-Max, Mean, MaxAbs, Robust, and Unit Vector
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
