"""
===============================================================================
MODEL TRAINING & HYPERPARAMETER TUNING PIPELINE
===============================================================================
Academic Goal:
Train and compare 3 fundamental classification algorithms:
  1. Logistic Regression (Linear baseline model)
  2. Random Forest (Ensemble decision trees with majority voting)
  3. XGBoost (Sequential gradient boosting decision trees)

LEARNING CONCEPTS FOR STUDENTS:
-------------------------------
MODEL 1: LOGISTIC REGRESSION
- WHAT: A linear model that computes a weighted sum of inputs and passes it through
        the sigmoid function to calculate class probability (between 0 and 1).
- WHY:  Functions as an easy, fast linear baseline. Shows if linear decision boundaries work.
- HOW:  z = w1*x1 + w2*x2 + ... + b;  Probability = 1 / (1 + exp(-z))

MODEL 2: RANDOM FOREST
- WHAT: An ensemble of multiple independent decision trees trained on random subsets of data.
- WHY:  Combines predictions from many trees to reduce overfitting and improve accuracy.
- HOW:  Each tree votes for a class. The majority vote determines the final predicted class.

MODEL 3: XGBOOST (EXREME GRADIENT BOOSTING)
- WHAT: A sequential ensemble technique where each new decision tree is built to correct
        the specific prediction errors (residuals) made by previous trees.
- WHY:  State-of-the-art accuracy on tabular space object data.
- HOW:  Tree 1 predicts -> Tree 2 learns from Tree 1 errors -> Tree 3 learns from Tree 2 errors.
===============================================================================
"""

import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from data_preprocessing import get_preprocessed_pipeline_data


def train_logistic_regression(X_train_scaled, y_train):
    """
    Train Model 1: Logistic Regression.
    Requires scaled inputs because linear solvers depend on feature magnitudes.
    """
    print("\n--- TRAINING MODEL 1: LOGISTIC REGRESSION ---")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"Logistic Regression trained in {train_time:.3f} seconds.")
    return model


def train_random_forest(X_train, y_train):
    """
    Train Model 2: Random Forest Classifier.
    Tree-based models do not require scaled inputs, but handle non-linear boundaries.
    """
    print("\n--- TRAINING MODEL 2: RANDOM FOREST ---")
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Random Forest (100 trees) trained in {train_time:.3f} seconds.")
    return model


def train_xgboost(X_train, y_train):
    """
    Train Model 3: XGBoost Classifier.
    Sequential gradient boosted trees optimizing logloss objective function.
    """
    print("\n--- TRAINING MODEL 3: XGBOOST ---")
    start_time = time.time()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"XGBoost trained in {train_time:.3f} seconds.")
    return model


def tune_best_model(X_train, y_train, best_model_type='random_forest'):
    """
    WHAT: Perform hyperparameter tuning using 3-fold cross-validation.
    WHY:  Hyperparameters are settings chosen BEFORE training (e.g. tree depth, number of trees).
          Tuning finds the optimal settings to maximize Recall and F1 score.
    HOW:  GridSearchCV tests combinations of parameters on validation folds and selects the best.
    """
    print(f"\n--- HYPERPARAMETER TUNING ({best_model_type.upper()}) ---")
    print("Testing hyperparameter combinations with 3-Fold Cross-Validation...")

    # Fast tuning on representative sample (10,000 rows) for quick iteration
    if len(X_train) > 10000:
        sample_idx = np.random.choice(len(X_train), 10000, replace=False)
        X_tune = X_train.iloc[sample_idx]
        y_tune = y_train.iloc[sample_idx]
    else:
        X_tune = X_train
        y_tune = y_train

    if best_model_type == 'random_forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [12, 18]
        }
    else:  # xgboost
        base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [5, 8],
            'learning_rate': [0.05, 0.1]
        }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='recall',  # Prioritize Recall for Space Debris hazard detection!
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_tune, y_tune)
    print(f"[TUNING COMPLETED] Best Hyperparameters Found: {grid_search.best_params_}")
    print(f"Best CV Recall Score: {grid_search.best_score_*100:.2f}%")

    # Re-fit best estimator on full training set
    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train)

    return best_estimator, grid_search.best_params_


def save_model_artifacts(model, scaler, feature_cols, params, save_path='models/final_model.pkl'):
    """
    WHAT: Serialize and save the trained model, feature names, scaler, and metadata to disk.
    WHY:  Training is done ONCE. In production / inference, the saved model file is loaded
          instantly to make predictions without retraining!
    HOW:  joblib.dump serializes Python objects into a single compressed binary file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'hyperparameters': params,
        'trained_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(artifact, save_path)
    print(f"\n[MODEL SAVED] Final model artifact successfully saved to '{save_path}'.")


def run_training_pipeline():
    """
    Executes complete model training, selection, tuning, and serialization pipeline.
    """
    # 1. Load Preprocessed Data
    data_dict = get_preprocessed_pipeline_data()
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    scaler = data_dict['scaler']
    feature_cols = data_dict['feature_cols']

    # 2. Train Initial 3 Models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # 3. Perform Hyperparameter Tuning on Top Candidate (Random Forest)
    tuned_rf_model, best_params = tune_best_model(X_train, y_train, best_model_type='random_forest')

    # 4. Save Final Model Artifact
    save_model_artifacts(
        model=tuned_rf_model,
        scaler=scaler,
        feature_cols=feature_cols,
        params=best_params,
        save_path='models/final_model.pkl'
    )

    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'tuned_model': tuned_rf_model,
        'data_dict': data_dict
    }


if __name__ == "__main__":
    run_training_pipeline()
