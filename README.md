<div align="center">

# 🛰️ Space Debris & Orbital Object Prediction
### *An Academic & Beginner-Friendly Machine Learning Masterclass*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-2A52BE?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/CelesTrak-70%2C580%20Objects-green?style=for-the-badge)](https://celestrak.org/pub/satcat.csv)
[![License](https://img.shields.io/badge/License-Academic%20Use-purple?style=for-the-badge)](#)

[Quickstart](#-quickstart--how-to-run) • [Dataset Architecture](#-dataset-architecture) • [ML Pipeline](#-machine-learning-pipeline) • [Model Benchmarks](#-model-comparison--evaluation) • [2-Min Pitch](#-explain-my-project-in-2-minutes)

---

</div>

> [!NOTE]
> **Academic & Learning Notice**: This project is built **strictly for academic and educational purposes** to demonstrate how real-world orbital tracking data is cleaned, preprocessed, engineered, modeled, evaluated, and deployed via an interactive Streamlit prediction interface.

> [!IMPORTANT]
> **Domain Classification Scope**: This project performs **Object-Type Classification** (`Class 1`: `Space Debris / Rocket Body` vs `Class 0`: `Payload Satellite`) based on physical Keplerian orbital parameters. It is an **academic classification model**, not a direct collision-risk or operational collision avoidance system.

---

## 📖 Table of Contents
- [✨ Project Overview](#-project-overview)
- [⚡ Quickstart & How to Run](#-quickstart--how-to-run)
- [📊 Dataset Architecture](#-dataset-architecture)
- [🎯 Understanding X and y](#-understanding-x-and-y)
- [🧹 Step-by-Step Data Cleaning](#-step-by-step-data-cleaning)
- [⚙️ Feature Engineering](#️-feature-engineering)
- [🛡️ Data Leakage Prevention](#️-data-leakage-prevention)
- [⚡ Why We Don't Retrain on Prediction](#-why-we-dont-retrain-on-prediction)
- [🔀 Train / Test Split](#-train--test-split)
- [📏 Feature Scaling (StandardScaler)](#-feature-scaling-standardscaler)
- [🤖 Machine Learning Algorithms](#-machine-learning-algorithms)
- [🔍 Single-Sample Prediction Trace](#-single-sample-prediction-trace)
- [📐 Evaluation Metrics & The Accuracy Trap](#-evaluation-metrics--the-accuracy-trap)
- [📈 Model Comparison & Project KPIs](#-model-comparison--project-kpis)
- [🎛️ Hyperparameter Tuning](#️-hyperparameter-tuning)
- [📚 Beginner ML Glossary](#-beginner-ml-glossary)
- [🎤 Explain My Project in 2 Minutes](#-explain-my-project-in-2-minutes)
- [✅ Can I Explain It? Checklist](#-can-i-explain-it-checklist)

---

## ✨ Project Overview

This project answers a fundamental space data question:
> *Given physical tracking measurements of an object orbiting Earth (Apogee, Perigee, Period, Inclination, Radar Size), can a Machine Learning model predict whether it is **Space Debris / Rocket Body Junk** (`Class 1`) or a **Payload Satellite** (`Class 0`)?*

### The End-to-End ML Flow:
```text
DATA ──► CLEAN ──► UNDERSTAND ──► FEATURES ──► X/y ──► TRAIN ──► PREDICT ──► EVALUATE ──► COMPARE
```

---

## ⚡ Quickstart & How to Run

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/space-debris-ml.git
cd space-debris-ml

# Install required Python dependencies
pip install -r requirements.txt
```

### 2. Run the Modular ML Pipeline
```bash
# Step 1: Clean raw data and compute Keplerian features
python src/data_preprocessing.py

# Step 2: Train Logistic Regression, Random Forest, XGBoost & Save Model
python src/train.py

# Step 3: Run evaluation benchmark & calculate 5 Project KPIs
python src/evaluate.py

# Step 4: Run single-sample inference test
python src/predict.py
```

### 3. Launch Interactive Streamlit Product Interface
```bash
py -m streamlit run app.py
```

---

## 📊 Dataset Architecture

- **Source**: [CelesTrak Satellite Catalog (SATCAT)](https://celestrak.org/pub/satcat.csv) (US Space Command / NORAD Data)
- **Total Rows (Samples)**: `70,580` cataloged space objects
- **Total Columns (Properties)**: `17` raw tracking fields

### What is One Row?
One row represents **one single physical object** orbiting Earth (e.g. SPUTNIK 1, SL-1 Rocket Body, Vanguard 1).

### What is One Column?
One column represents **one specific property or measurement** of that object.

```text
Object A (Row / Sample)
│
├── Period        (Orbital duration in minutes)
├── Inclination   (Tilt angle relative to Equator in degrees)
├── Apogee        (Highest altitude above Earth in km)
├── Perigee       (Lowest altitude above Earth in km)
└── RCS           (Radar Cross Section size in m²)
```

---

## 🎯 Understanding X and y

```text
                 X (Input Features Matrix)
        ┌─────────────────────────────────────────┐
        │ Period (minutes)                        │
        │ Inclination (degrees)                   │
        │ Apogee Altitude (km)                    │
        │ Perigee Altitude (km)                   │
        │ Radar Cross Section - RCS (m²)          │
        │ Mean Altitude (km) [Engineered]         │
        │ Eccentricity [Engineered]               │
        │ Orbital Velocity (km/s) [Engineered]    │
        └─────────────────────────────────────────┘
                            │
                            ▼
                        ML MODEL
                            │
                            ▼
                 y (Target Answer Vector)
        ┌─────────────────────────────────────────┐
        │ 0 = Payload Satellite                   │
        │ 1 = Space Debris / Rocket Body          │
        └─────────────────────────────────────────┘
```

- **$X$ (Features)**: What we give the model (input information).
- **$y$ (Target)**: The answer we want the model to learn to predict.

---

## 🧹 Step-by-Step Data Cleaning

### Step 1: Filter Unknown Objects
- **BEFORE**: Dataset contains `DEB` (Debris), `PAY` (Payload), `R/B` (Rocket Body), and `UNK` (Unknown - 165 objects).
- **PROBLEM**: `UNK` objects lack confirmed labels.
- **WHAT WE DO**: Exclude `UNK`. Map `DEB` + `R/B` $\rightarrow 1$ (`Space Debris / Rocket Body`), and `PAY` $\rightarrow 0$ (`Payload Satellite`).
- **WHY**: Machine Learning models need clean ground-truth target labels ($y$).

---

### Step 2: Remove Missing Orbital Parameters
- **BEFORE**: 2,054 objects are missing Period, Apogee, or Perigee values.
- **PROBLEM**: Without Apogee and Perigee, we cannot calculate orbital height or velocity!
- **WHAT WE DO**: Remove rows missing fundamental orbital parameters (`dropna`).
- **WHY**: Trajectory calculations require complete physical measurements.

---

### Step 3: Impute Missing Radar Cross Section (RCS)
- **BEFORE**: RCS (radar size) has missing values.
- **PROBLEM**: We don't want to throw away 37,000 valid orbital rows just because radar size was missing!
- **WHAT WE DO**: Convert RCS to numbers and fill missing values with the **Median** ($0.0645\text{ m}^2$).
- **WHAT IS MEDIAN?**: The exact middle value when numbers are ordered from smallest to largest. Unlike the mean (average), the median is not distorted by extreme outliers (like huge space stations).

---

## ⚙️ Feature Engineering

Instead of looking at Apogee and Perigee separately, we calculate 4 physical Keplerian features:

| Feature Name | Formula / Derivation | Physical Meaning & Rationale |
|---|---|---|
| **Mean Altitude** | $\frac{\text{Apogee} + \text{Perigee}}{2}$ ($\text{km}$) | Average height of the orbit above Earth's surface. |
| **Eccentricity** | $\frac{\text{Apogee} - \text{Perigee}}{\text{Apogee} + \text{Perigee} + 2 \times 6371.0}$ | Measures orbital roundness ($0 = \text{circular orbit}$, values near $1 = \text{stretched ellipse}$). |
| **Semi-Major Axis** | $\text{Mean Altitude} + 6371.0\text{ km}$ | Distance from Earth's center to the furthest point of orbit. |
| **Orbital Velocity** | $\sqrt{\frac{398600.4418}{\text{Semi-Major Axis}}}$ ($\text{km/s}$) | Approximate orbital speed based on Keplerian gravitational mechanics. |

---

## 🛡️ Data Leakage Prevention

> [!CAUTION]
> **Data Leakage** occurs when features contain future knowledge or direct label shortcuts that will not be available when predicting new unseen data.

### Columns Removed to Prevent Leakage:
- `OPS_STATUS_CODE`: Operational status `D` (Decayed) directly reveals if an object broke up into debris!
- `DECAY_DATE`: Directly reveals if an object re-entered Earth's atmosphere.
- `OBJECT_NAME`: Names contain strings like `"SL-1 R/B"` or `"DEB"`. A model could simply read string patterns instead of learning orbital physics!

---

## ⚡ Why We Don't Retrain on Prediction

```text
TRAINING PHASE (Done ONCE)
Dataset (54,000+ objects) ──► ML Model ──► Learns Tree Split Rules ──► Saved to 'models/final_model.pkl'

PREDICTION PHASE (Done in App)
User Input ──► Load 'final_model.pkl' ──► Evaluate Split Rules ──► Output Result (< 5 ms)
```

- **Training**: The model analyzes 54,000+ historical satellite tracks to learn split boundaries (takes seconds/minutes).
- **Prediction (Inference)**: The model loads pre-saved weights from disk and evaluates a single vector (takes under 5 milliseconds).
- **Key Point**: We train the model **once**. We do **not** retrain it every time a user enters a new object!

---

## 🔀 Train / Test Split

Suppose we have 100 orbital objects:
- **80 Objects (80%) $\rightarrow$ Training Set**: Used by the model to learn patterns.
- **20 Objects (20%) $\rightarrow$ Testing Set**: Kept unseen to test how well the model works on new data.

```text
Total Dataset (68,361 objects)
          │
          ├─── 80% Training Set (54,688 objects)  ──► Model Learns Here
          │
          └─── 20% Unseen Test Set (13,673 objects) ──► Model Tested Here
```

> **Why keep test data unseen?**
> Testing on the exact same data used for training would be like giving a student the exam answers before the test! It measures memory, not true understanding.

---

## 📏 Feature Scaling (StandardScaler)

Look at our raw input values:
- `Period` $\approx 96.0$
- `Apogee` $\approx 938.0$
- `RCS_NUM` $\approx 0.0645$
- `Eccentricity` $\approx 0.052$

These values exist on completely different scales! Linear models like **Logistic Regression** can get confused because large numbers (Apogee = 938) dominate small numbers (RCS = 0.0645).

### What does StandardScaler do?
It scales features so that $\text{Mean} = 0$ and $\text{Std} = 1$.

> **Which models need scaling?**
> - **Logistic Regression**: **Requires scaling** because gradient optimization depends on feature scales.
> - **Tree Models (Random Forest, XGBoost)**: **Do not require scaling** because decision trees split features independently (`Apogee > 500`).

---

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression (Linear Baseline)
Logistic Regression tries to draw a straight linear decision boundary between two classes.

```text
                  Space Debris (Class 1)
                     ↑
              ●  ●  ●  ●  ●
           ●  ●  ●  ●  ●
--------------------------------------- Decision Boundary
        ○  ○  ○  ○  ○
     ○  ○  ○  ○  ○
                     ↓
             Payload (Class 0)
```

1. Computes weighted sum: $z = w_1 \cdot \text{Period} + w_2 \cdot \text{Apogee} + \dots + b$
2. Sigmoid function: $\text{Probability} = \frac{1}{1 + e^{-z}}$
3. Output: Probability $\ge 0.5 \rightarrow \text{Class 1 (Debris)}$, else $\text{Class 0 (Payload)}$.

---

### 2. Decision Trees (Before Random Forest)
A single Decision Tree makes decisions by asking step-by-step threshold questions:

```text
                  Is Perigee < 500 km?
                       /       \
                     YES        NO
                     /           \
         Is RCS < 0.1 m²?       Payload (Class 0)
             /       \
           YES        NO
           /           \
  Debris (Class 1)   Payload (Class 0)
```

---

### 3. Random Forest (Ensemble Majority Voting)
A single tree can make mistakes. A **Random Forest** builds 100 independent decision trees and combines their predictions via **majority vote**:

```text
Tree 1  ──►  Predicts: Debris (1)
Tree 2  ──►  Predicts: Debris (1)
Tree 3  ──►  Predicts: Payload (0)
Tree 4  ──►  Predicts: Debris (1)
Tree 5  ──►  Predicts: Debris (1)

Majority Vote Result ──► Class 1 (Debris)
```

---

### 4. XGBoost (Extreme Gradient Boosting)
Unlike Random Forest (where trees are independent), **XGBoost builds trees sequentially**. Each new tree is specifically trained to correct errors made by earlier trees!

```text
Tree 1 makes predictions
          │
          ▼
    Makes mistakes on 1,000 samples
          │
          ▼
Tree 2 focuses specifically on those 1,000 mistakes
          │
          ▼
Tree 3 focuses on remaining errors
          │
          ▼
Final Combined Prediction
```

---

## 🔍 Single-Sample Prediction Trace

```text
User Input:
- Apogee = 938.0 km, Perigee = 466.0 km, Period = 96.19 min, Inclination = 65.10°, RCS = 0.08 m²
         ↓
Feature Engineering:
- Mean Altitude = (938 + 466) / 2 = 702.0 km
- Eccentricity = (938 - 466) / (938 + 466 + 2 * 6371) = 0.0335
- Semi-Major Axis = 702 + 6371 = 7073.0 km
- Velocity = sqrt(398600.4418 / 7073) = 7.507 km/s
         ↓
Pass 8-Feature Vector to Saved Model ('models/final_model.pkl')
         ↓
Random Forest 100 Trees Vote
         ↓
Output Probability: 92.28% Debris, 7.72% Payload
         ↓
Final Prediction: Class 1 (Space Debris / Rocket Body)
```

---

## 📐 Evaluation Metrics & The Accuracy Trap

### Tiny Example (4 Test Objects):
| Object | Actual True Class | Model Predicted Class | Result Type |
|---|---|---|---|
| Object 1 | Debris (1) | Debris (1) | **True Positive (TP)** |
| Object 2 | Debris (1) | Payload (0) | **False Negative (FN)** ⚠️ (Missed Debris!) |
| Object 3 | Payload (0) | Debris (1) | **False Positive (FP)** (False Alarm) |
| Object 4 | Payload (0) | Payload (0) | **True Negative (TN)** |

- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{Total}}$ (Percentage of correct predictions).
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (When model says Debris, how often is it right?).
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Out of all actual debris, how many did we catch?).
- **F1 Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic balance).

### The Accuracy Trap
Suppose a dataset has **95 Payload satellites** and **5 Debris objects**.
A lazy model predicting "Payload" for everything gets **95% Accuracy**, but **0% Recall for Debris**! It misses 100% of debris objects!
That is why evaluating Recall and F1 Score is critical.

---

## 📈 Model Comparison & Project KPIs

Evaluated on **13,673 unseen test objects**:

### 📊 Model Comparison Table
| Model | Accuracy | Precision | Recall (Debris Rate) | F1-Score | ROC-AUC | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.7359 | 0.7325 | 0.8850 | 0.8016 | 0.7220 | **0.747 ms** |
| **Random Forest (Default)** | **0.9291** | **0.9219** | 0.9640 | **0.9425** | **0.9794** | 43.890 ms |
| **XGBoost** | 0.9236 | 0.9162 | 0.9613 | 0.9382 | 0.9788 | **5.694 ms** |
| **Random Forest (Tuned)** | 0.9234 | 0.9134 | **0.9643** | 0.9382 | 0.9776 | 43.434 ms |

### 🎯 Project KPI Results Table
| KPI Name | Type | Measurement | Target | Actual | Status & Interpretation |
|---|---|---|---:|---:|---|
| **Debris Hazard Recall** | Business | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | $\ge 85.0\%$ | **96.43%** | **PASSED** — Caught 96.43% of space debris objects. |
| **Recall Score** | ML Metric | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | $\ge 85.0\%$ | **96.43%** | **PASSED** — Minimizes dangerous False Negatives. |
| **F1 Score** | ML Metric | $2 \times \frac{P \times R}{P + R}$ | $\ge 85.0\%$ | **93.82%** | **PASSED** — Strong balance between precision and recall. |
| **Data Completeness** | Data Quality | $\frac{\text{Valid Cells}}{\text{Total Cells}} \times 100$ | $\ge 80.0\%$ | **88.16%** | **PASSED** — High quality clean input dataset. |
| **Prediction Latency** | Engineering | Time per sample | $< 10.0\text{ ms}$ | **5.694 ms** | **PASSED** — Single sample XGBoost inference in 5.69 ms. |

---

## 🎛️ Hyperparameter Tuning

- **Parameters**: Learned automatically during training (e.g. weights $w_1, w_2$ or tree split values).
- **Hyperparameters**: Settings chosen by us **before** training (e.g. `n_estimators`, `max_depth`).
- **GridSearchCV**: Tests combinations of settings on validation folds and selects the best configuration.

---

## 📚 Beginner ML Glossary

<details>
<summary><b>Click to expand the 24-Term ML Glossary</b></summary>

<br>

1. **Dataset**: A structured table of rows and columns.
2. **Row (Sample)**: One single object record in the dataset.
3. **Column (Feature)**: A property or measurement of an object.
4. **Feature ($X$)**: Input measurement given to an ML model.
5. **Target ($y$)**: The answer the model learns to predict.
6. **$X$ Matrix**: Table containing all input feature columns.
7. **$y$ Vector**: Column containing ground-truth target answers.
8. **Training**: The process where a model fits rules on training samples.
9. **Testing**: Evaluating model predictions on unseen samples.
10. **Model**: An algorithm that maps inputs $X$ to outputs $y$.
11. **Parameter**: Internal weights learned by the model during training.
12. **Hyperparameter**: Settings configured before training (`n_estimators`).
13. **Prediction**: The output generated by a trained model for an input.
14. **Inference**: Running predictions using saved model weights.
15. **Classification**: Predicting discrete categories (0 vs 1).
16. **Probability**: Confidence score between 0.0 and 1.0.
17. **Accuracy**: Percentage of total predictions that are correct.
18. **Precision**: Out of positive predictions, how many were correct.
19. **Recall**: Out of actual positive samples, how many were caught.
20. **F1 Score**: Harmonic mean of Precision and Recall.
21. **Overfitting**: Memorizing training data perfectly while failing on test data.
22. **Data Leakage**: Features exposing future knowledge or label shortcuts.
23. **Feature Engineering**: Creating domain features using mathematical transformations.
24. **Feature Scaling**: Adjusting feature values to a standard scale ($\text{Mean}=0, \text{Std}=1$).

</details>

---

## 🎤 Explain My Project in 2 Minutes

> *"Professor, for my project I built a Machine Learning system to classify tracked objects orbiting Earth into **Space Debris / Rocket Bodies** versus **Payload Satellites**.
>
> I used real tracking data from CelesTrak containing over 70,000 cataloged objects. I cleaned missing orbital values, median-imputed radar cross-section sizes, and engineered physical Keplerian features like Mean Altitude, Eccentricity, and Orbital Velocity to help the model learn gravitational mechanics.
>
> I defined $X$ as the orbital features and $y$ as the binary target ($1$ for Debris/Rocket Body, $0$ for Payload). I removed columns like operational status and decay date to prevent data leakage.
>
> I split the dataset into $80\%$ Training and $20\%$ Unseen Testing. I trained 3 algorithms: **Logistic Regression** as a linear baseline ($73.6\%$ accuracy), **Random Forest** ($92.9\%$ accuracy), and **XGBoost** ($92.4\%$ accuracy).
>
> In aerospace safety, missing a piece of debris (False Negative) is much more dangerous than a false alarm. That's why I prioritized **Recall**. Random Forest achieved a **$96.4\%$ Recall Rate**, catching almost all debris objects.
>
> Finally, I saved the trained model parameters into `models/final_model.pkl` and created a Streamlit interface. When a user enters orbital numbers, the app loads the saved model and generates a prediction in under $5\text{ ms}$ without retraining."*

---

## ✅ Can I Explain It? Checklist

- [x] I know what one row represents (one cataloged orbital object).
- [x] I know what each feature means (Apogee, Perigee, Period, Inclination, RCS).
- [x] I know what $X$ means (inputs given to the model).
- [x] I know what $y$ means (target answer: 1 = Debris/Rocket Body, 0 = Payload).
- [x] I know why we clean data (remove UNK, handle missing values, impute RCS).
- [x] I know why we split the dataset (80% train, 20% unseen test to check generalization).
- [x] I know what training means (model fits rules on training samples).
- [x] I know what inference means (evaluating new inputs using saved weights).
- [x] I know how Logistic Regression works (weighted sum + Sigmoid function).
- [x] I know what a Decision Tree is (step-by-step threshold questions).
- [x] I know how Random Forest works (ensemble of 100 trees voting by majority).
- [x] I know the basic idea of XGBoost (trees built sequentially to correct errors).
- [x] I know what overfitting means (memorizing train data while failing on test data).
- [x] I know what data leakage means (features revealing future answers).
- [x] I know Accuracy (correct predictions / total predictions).
- [x] I know Precision (correct positive predictions / total positive predictions).
- [x] I know Recall (caught positive samples / total actual positive samples).
- [x] I know F1 Score (harmonic mean balancing Precision and Recall).
- [x] I understand the confusion matrix (TP, TN, FP, FN).
- [x] I understand why the final model was selected (Random Forest / XGBoost high Recall).
- [x] I understand what happens when Predict is clicked (load saved `.pkl`, compute features, predict).
- [x] I know that prediction does NOT retrain the model.

---

<div align="center">
  <b>Built for Academic & Learning Purposes • Powered by CelesTrak SATCAT Data</b>
</div>
