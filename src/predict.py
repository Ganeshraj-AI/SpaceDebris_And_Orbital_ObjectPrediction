"""
===============================================================================
SINGLE-SAMPLE INFERENCE & PREDICTION MODULE
===============================================================================
Academic Goal:
Demonstrate how a saved machine learning model is loaded and used to make
predictions on new, unseen orbital inputs WITHOUT retraining the model.

LEARNING CONCEPT:
-----------------
WHAT:  Inference / Prediction.
WHY:   Model training is done ONCE. Saved weights (final_model.pkl) are loaded
       instantly to make predictions on new input vectors.
HOW:   User Input -> Feature Engineering -> Model.predict_proba -> Output

CONCEPTUAL CLARIFICATION:
-------------------------
Target Categories:
  - Class 1: Space Debris / Rocket Body (Uncontrolled orbital object)
  - Class 0: Payload Satellite (Active or inactive spacecraft)

This is an OBJECT-TYPE CLASSIFICATION model based on orbital characteristics,
NOT a direct collision-risk or operational collision avoidance system!
===============================================================================
"""

import os
import time
import joblib
import pandas as pd
import numpy as np


def load_trained_model(model_path='models/final_model.pkl'):
    """
    Load saved model artifact from disk.
    Avoids retraining the model on every prediction request.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model file not found at '{model_path}'. Please run 'src/train.py' first!"
        )
    artifact = joblib.load(model_path)
    return artifact


def predict_space_object(period, inclination, apogee, perigee, rcs_num=0.0645, model_path='models/final_model.pkl'):
    """
    Takes raw orbital parameters, computes engineered features, and predicts object category.
    """
    # 1. Load Saved Model Artifact (No retraining!)
    artifact = load_trained_model(model_path)
    model = artifact['model']
    feature_cols = artifact['feature_cols']

    # 2. Feature Engineering on Input Vector
    EARTH_RADIUS_KM = 6371.0
    EARTH_MU = 398600.4418

    altitude_mean = (apogee + perigee) / 2.0
    eccentricity = (apogee - perigee) / (apogee + perigee + 2.0 * EARTH_RADIUS_KM)
    semi_major_axis = altitude_mean + EARTH_RADIUS_KM
    velocity_km_s = np.sqrt(EARTH_MU / semi_major_axis)
    apogee_perigee_ratio = (apogee + EARTH_RADIUS_KM) / (perigee + EARTH_RADIUS_KM)

    input_dict = {
        'PERIOD': [float(period)],
        'INCLINATION': [float(inclination)],
        'APOGEE': [float(apogee)],
        'PERIGEE': [float(perigee)],
        'ALTITUDE_MEAN': [float(altitude_mean)],
        'ECCENTRICITY': [float(eccentricity)],
        'VELOCITY_KM_S': [float(velocity_km_s)],
        'RCS_NUM': [float(rcs_num)]
    }

    input_df = pd.DataFrame(input_dict)[feature_cols]

    # 3. Measure Prediction Latency
    start_time = time.perf_counter()
    pred_class = int(model.predict(input_df)[0])
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        debris_prob = float(probabilities[1])
        payload_prob = float(probabilities[0])
    else:
        debris_prob = 1.0 if pred_class == 1 else 0.0
        payload_prob = 1.0 - debris_prob

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Accurate Object-Type Category Labels
    if pred_class == 1:
        category_label = "Space Debris / Rocket Body"
        confidence = debris_prob * 100.0
    else:
        category_label = "Payload Satellite"
        confidence = payload_prob * 100.0

    return {
        'prediction_class': pred_class,
        'category_label': category_label,
        'confidence_percent': confidence,
        'debris_probability': debris_prob,
        'payload_probability': payload_prob,
        'latency_ms': latency_ms,
        'features_used': input_dict
    }


if __name__ == "__main__":
    print("Testing Prediction Module on Sample Low Earth Orbit Object...")
    res = predict_space_object(period=96.19, inclination=65.10, apogee=938.0, perigee=466.0, rcs_num=0.08)
    print("\n--- PREDICTION RESULT ---")
    print(f"Predicted Class:      {res['prediction_class']} ({res['category_label']})")
    print(f"Debris Probability:   {res['debris_probability']*100:.2f}%")
    print(f"Payload Probability:  {res['payload_probability']*100:.2f}%")
    print(f"Inference Latency:    {res['latency_ms']:.3f} ms")
