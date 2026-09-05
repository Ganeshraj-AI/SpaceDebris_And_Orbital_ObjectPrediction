"""
===============================================================================
MODEL EVALUATION & COMPARISON MODULE
===============================================================================
Academic Goal:
Evaluate Logistic Regression, Random Forest, and XGBoost on unseen test data.
Calculate evaluation metrics, confusion matrix components (TP, TN, FP, FN),
measure prediction latency, and report all required Project KPIs.

LEARNING CONCEPTS FOR STUDENTS:
-------------------------------
1. ACCURACY:
   - What: Ratio of correct predictions to total predictions.
   - Formula: (TP + TN) / (TP + TN + FP + FN)

2. PRECISION:
   - What: Out of all objects PREDICTED as space debris, how many were actually debris?
   - Formula: TP / (TP + FP)

3. RECALL (SENSITIVITY) — MOST CRITICAL METRIC FOR SPACE HAZARD SAFETY!
   - What: Out of all ACTUAL space debris objects, how many did the model identify?
   - Formula: TP / (TP + FN)
   - Why it matters: A FALSE NEGATIVE (FN) means an un-tracked space debris hazard was
     misclassified as a safe payload satellite! Missing debris poses collision risks!

4. F1-SCORE:
   - What: The harmonic mean of Precision and Recall.
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)

5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
   - What: Measures how well the model separates Positive (Debris) from Negative (Payload)
     across all possible classification probability thresholds (0.0 to 1.0).
   - Score: 1.0 = Perfect separation, 0.5 = Random guessing.
===============================================================================
"""

import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data_preprocessing import get_preprocessed_pipeline_data
from train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    tune_best_model
)


def measure_prediction_latency(model, X_sample, is_scaled=False, scaler=None):
    """
    WHAT: Measure average inference execution time per single prediction in milliseconds.
    WHY:  Engineering KPI measuring how fast the model generates a risk prediction.
    HOW:  Run model.predict 100 times on a single sample and calculate average duration.
    """
    sample = X_sample.iloc[[0]]
    if is_scaled and scaler is not None:
        sample = scaler.transform(sample)

    # Warmup
    _ = model.predict(sample)

    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        _ = model.predict(sample)
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = total_time_ms / iterations
    return avg_latency_ms


def evaluate_single_model(model_name, model, X_test, y_test, is_scaled=False, scaler=None):
    """
    Computes all standard classification metrics, confusion matrix, and latency for a given model.
    """
    if is_scaled and scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    latency_ms = measure_prediction_latency(model, X_test, is_scaled=is_scaled, scaler=scaler)

    results = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'Latency (ms)': latency_ms,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

    return results


def run_full_evaluation():
    """
    Loads data once, trains all 3 models, evaluates them on unseen test set, prints comparison table & KPI table.
    """
    print("===============================================================================")
    print("             EXPERT MODEL EVALUATION & COMPARISON BENCHMARK")
    print("===============================================================================")

    # 1. Load Preprocessed Data ONCE
    data_dict = get_preprocessed_pipeline_data()
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    scaler = data_dict['scaler']

    # 2. Train Models directly using pre-split data
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    tuned_rf_model, best_params = tune_best_model(X_train, y_train, best_model_type='random_forest')

    # 3. Evaluate Each Model
    eval_results = []
    eval_results.append(evaluate_single_model("Logistic Regression", lr_model, X_test, y_test, is_scaled=True, scaler=scaler))
    eval_results.append(evaluate_single_model("Random Forest (Default)", rf_model, X_test, y_test, is_scaled=False))
    eval_results.append(evaluate_single_model("XGBoost", xgb_model, X_test, y_test, is_scaled=False))
    eval_results.append(evaluate_single_model("Random Forest (Tuned)", tuned_rf_model, X_test, y_test, is_scaled=False))

    results_df = pd.DataFrame(eval_results)

    # 4. Display Model Comparison Table
    print("\n" + "="*85)
    print("                        MODEL COMPARISON TABLE (UNSEEN TEST DATA)")
    print("="*85)
    fmt_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Latency (ms)']].copy()
    fmt_df['Accuracy'] = fmt_df['Accuracy'].map('{:.4f}'.format)
    fmt_df['Precision'] = fmt_df['Precision'].map('{:.4f}'.format)
    fmt_df['Recall'] = fmt_df['Recall'].map('{:.4f}'.format)
    fmt_df['F1-Score'] = fmt_df['F1-Score'].map('{:.4f}'.format)
    fmt_df['ROC-AUC'] = fmt_df['ROC-AUC'].map('{:.4f}'.format)
    fmt_df['Latency (ms)'] = fmt_df['Latency (ms)'].map('{:.3f}'.format)
    print(fmt_df.to_string(index=False))

    # 5. Detailed Confusion Matrix Breakdown for Best Model
    best_row = results_df.sort_values(by='Recall', ascending=False).iloc[0]
    print("\n" + "="*85)
    print(f"       CONFUSION MATRIX BREAKDOWN FOR SELECTED MODEL: [{best_row['Model']}]")
    print("="*85)
    print(f"  True Positives  (TP - Correctly identified Debris):   {best_row['TP']}")
    print(f"  True Negatives  (TN - Correctly identified Payload):  {best_row['TN']}")
    print(f"  False Positives (FP - False Alarms):                  {best_row['FP']}")
    print(f"  False Negatives (FN - MISSED DEBRIS HAZARDS):         {best_row['FN']}")
    print("-" * 85)
    print(f"  CRITICAL ANALYSIS: The model missed only {best_row['FN']} out of {best_row['TP']+best_row['FN']} space debris objects,")
    print(f"  achieving a High Debris Hazard Recall of {best_row['Recall']*100:.2f}%.")

    # 6. Build KPI Table
    completeness_kpi = (data_dict['df_clean'].notnull().sum().sum() / data_dict['df_clean'].size) * 100.0
    
    kpis = [
        {
            'KPI Name': 'Hazard Detection Rate',
            'Type': 'Business',
            'Formula / Measurement': 'TP / (TP + FN)',
            'Target': '>= 85.0%',
            'Actual': f"{best_row['Recall']*100:.2f}%",
            'Interpretation': 'Measures proportion of space debris hazards caught.'
        },
        {
            'KPI Name': 'Recall Score',
            'Type': 'ML Metric',
            'Formula / Measurement': 'TP / (TP + FN)',
            'Target': '>= 85.0%',
            'Actual': f"{best_row['Recall']*100:.2f}%",
            'Interpretation': 'High recall minimizes critical false negative risks.'
        },
        {
            'KPI Name': 'F1 Score',
            'Type': 'ML Metric',
            'Formula / Measurement': '2 * (P * R) / (P + R)',
            'Target': '>= 85.0%',
            'Actual': f"{best_row['F1-Score']*100:.2f}%",
            'Interpretation': 'Harmonic balance between precision and recall.'
        },
        {
            'KPI Name': 'Data Completeness',
            'Type': 'Data Quality',
            'Formula / Measurement': 'Valid Values / Total Expected',
            'Target': '>= 80.0%',
            'Actual': f"{completeness_kpi:.2f}%",
            'Interpretation': 'Ensures high quality input before modeling.'
        },
        {
            'KPI Name': 'Prediction Latency',
            'Type': 'Engineering',
            'Formula / Measurement': 'Milliseconds per sample',
            'Target': '< 10.0 ms',
            'Actual': f"{best_row['Latency (ms)']:.3f} ms",
            'Interpretation': 'Fast single-sample inference time.'
        }
    ]

    kpi_df = pd.DataFrame(kpis)
    print("\n" + "="*85)
    print("                           PROJECT KPI RESULTS TABLE")
    print("="*85)
    print(kpi_df.to_string(index=False))

    return results_df, kpi_df


if __name__ == "__main__":
    run_full_evaluation()
