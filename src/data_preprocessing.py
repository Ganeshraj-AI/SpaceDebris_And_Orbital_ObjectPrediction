"""
===============================================================================
DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE
===============================================================================
Academic Goal:
This module handles reading raw space object data, cleaning messy records,
creating physical orbital features, preventing data leakage, and splitting the
dataset into training and testing sets.

LEARNING CONCEPT:
-----------------
WHAT: We transform messy raw CSV data into clean, structured numerical matrices (X and y).
WHY:  Machine Learning algorithms cannot understand raw text or missing values.
      They require clean, scaled numerical inputs.
HOW:  1. Filter target classes (Space Debris/Rocket Body vs Payload Satellite).
      2. Drop missing/invalid orbital elements.
      3. Impute missing physical parameters (Radar Cross Section - RCS).
      4. Calculate Keplerian orbital parameters (Mean Altitude, Eccentricity, Velocity).
      5. Remove leakage columns that reveal future outcomes.
      6. Split into 80% Training and 20% Unseen Testing sets with stratification.
===============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(data_path='data/dataset.csv'):
    """
    WHAT: Load the raw CSV dataset into a pandas DataFrame.
    WHY:  Pandas provides efficient tabular data structures and data manipulation methods.
    HOW:  pd.read_csv reads the file and parses headers into column names.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data download script first.")
    df = pd.read_csv(data_path)
    print(f"[DATA LOAD] Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_data(df):
    """
    WHAT: Filter valid categories, define binary target, and handle missing values & duplicates.
    WHY:  'UNK' (Unknown) objects lack verified labels. Missing orbital values (Apogee/Perigee)
          prevent computing trajectory features.
    HOW:  - Keep only 'DEB' (Debris), 'R/B' (Rocket Body), and 'PAY' (Payload Satellite).
          - Define Target `is_space_debris`: 1 = Debris/Rocket Body (Junk), 0 = Payload.
          - Remove exact duplicates and invalid physical orbits (negative period/apogee/perigee).
    """
    print("\n--- STEP 1: DATA CLEANING ---")
    
    # 1. Filter target object types (Exclude UNK - 165 rows)
    initial_rows = len(df)
    valid_types = ['DEB', 'R/B', 'PAY']
    df_clean = df[df['OBJECT_TYPE'].isin(valid_types)].copy()
    print(f"Filter Object Types: Kept {len(df_clean)} / {initial_rows} rows (Removed UNK objects).")

    # 2. Define Binary Classification Target
    # 1 = Space Debris / Rocket Body (Uncontrolled Hazard Junk)
    # 0 = Operational / Non-operational Payload (Satellite)
    df_clean['is_space_debris'] = df_clean['OBJECT_TYPE'].apply(lambda x: 1 if x in ['DEB', 'R/B'] else 0)

    # 3. Handle Duplicate Rows
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Remove Duplicates: Dropped {dup_count} exact duplicate rows.")
    else:
        print("Remove Duplicates: No exact duplicate rows found.")

    # 4. Handle Missing Orbital Parameters
    # Objects missing Period, Inclination, Apogee, or Perigee cannot be used for orbital trajectory analysis.
    orbital_cols = ['PERIOD', 'INCLINATION', 'APOGEE', 'PERIGEE']
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=orbital_cols).copy()
    print(f"Missing Orbital Elements: Dropped {before_drop - len(df_clean)} rows missing Apogee/Perigee/Period.")

    # 5. Remove Invalid / Impossible Orbital Values
    # In physical space mechanics, altitude and orbital period must be positive.
    df_clean = df_clean[(df_clean['PERIOD'] > 0) & (df_clean['APOGEE'] >= 0) & (df_clean['PERIGEE'] >= 0)].copy()

    # 6. Impute Missing Radar Cross Section (RCS)
    # RCS measures object size in square meters. Convert to numeric and impute median for missing values.
    df_clean['RCS_NUM'] = pd.to_numeric(df_clean['RCS'], errors='coerce')
    median_rcs = df_clean['RCS_NUM'].median()
    df_clean['RCS_NUM'] = df_clean['RCS_NUM'].fillna(median_rcs)
    print(f"Impute RCS: Missing Radar Cross Section values imputed with median ({median_rcs:.4f} m²).")

    # Calculate Data Completeness Quality Metric KPI
    total_cells = df_clean.size
    non_null_cells = df_clean.notnull().sum().sum()
    completeness_kpi = (non_null_cells / total_cells) * 100.0
    print(f"[DATA QUALITY KPI] Data Completeness: {completeness_kpi:.2f}% non-missing values across all fields.")

    return df_clean


def engineer_features(df):
    """
    WHAT: Calculate physical Keplerian orbital parameters from raw orbital parameters.
    WHY:  Raw Apogee and Perigee alone do not directly capture orbital shape (Eccentricity),
          average altitude, or orbital velocity. Creating domain-specific features helps
          tree-based and linear models discover physical patterns easily.
    HOW:  Mathematical Transformations:
          - Mean Altitude = (Apogee + Perigee) / 2  (km)
          - Eccentricity = (Apogee - Perigee) / (Apogee + Perigee + 2 * Earth_Radius)
          - Semi-Major Axis (a) = Mean Altitude + 6371.0  (km)
          - Orbital Speed (v) = sqrt(GM / a) = sqrt(398600.4418 / a)  (km/s)
          - Apogee/Perigee Ratio = (Apogee + 6371) / (Perigee + 6371)
    """
    print("\n--- STEP 2: FEATURE ENGINEERING ---")
    df_fe = df.copy()

    EARTH_RADIUS_KM = 6371.0
    EARTH_MU = 398600.4418  # Earth Gravitational Parameter (km^3/s^2)

    # 1. Mean Altitude (Average height above Earth's surface)
    df_fe['ALTITUDE_MEAN'] = (df_fe['APOGEE'] + df_fe['PERIGEE']) / 2.0

    # 2. Eccentricity (Measures how elongated an orbit is: 0 = perfect circle, near 1 = elongated ellipse)
    df_fe['ECCENTRICITY'] = (df_fe['APOGEE'] - df_fe['PERIGEE']) / (df_fe['APOGEE'] + df_fe['PERIGEE'] + 2.0 * EARTH_RADIUS_KM)

    # 3. Semi-Major Axis (Distance from Earth's center to furthest point of orbit)
    df_fe['SEMI_MAJOR_AXIS'] = df_fe['ALTITUDE_MEAN'] + EARTH_RADIUS_KM

    # 4. Approximate Orbital Velocity (Vis-Viva / Circular speed approximation in km/s)
    df_fe['VELOCITY_KM_S'] = np.sqrt(EARTH_MU / df_fe['SEMI_MAJOR_AXIS'])

    # 5. Apogee to Perigee Ratio
    df_fe['APOGEE_PERIGEE_RATIO'] = (df_fe['APOGEE'] + EARTH_RADIUS_KM) / (df_fe['PERIGEE'] + EARTH_RADIUS_KM)

    print("Created Engineered Features: ALTITUDE_MEAN, ECCENTRICITY, SEMI_MAJOR_AXIS, VELOCITY_KM_S, APOGEE_PERIGEE_RATIO.")
    return df_fe


def select_features_and_target(df):
    """
    WHAT: Select feature matrix X and target vector y while removing data leakage columns.
    WHY:  Data leakage occurs when features contain future information or direct label shortcuts
          (like operational status or satellite names containing 'DEB').
    HOW:  Explicitly keep only physical orbital parameters. Drop name, ID, operational status, decay date.
    """
    print("\n--- STEP 3: FEATURE & TARGET DEFINITION ---")
    
    feature_columns = [
        'PERIOD',          # Orbital Period in minutes
        'INCLINATION',     # Orbital Inclination in degrees
        'APOGEE',          # Apogee altitude in km
        'PERIGEE',         # Perigee altitude in km
        'ALTITUDE_MEAN',    # Mean altitude in km
        'ECCENTRICITY',    # Orbital eccentricity (0 to 1)
        'VELOCITY_KM_S',   # Orbital speed in km/s
        'RCS_NUM'          # Radar Cross Section size in m^2
    ]
    
    target_column = 'is_space_debris'

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    print(f"X (Features Matrix Shape): {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"y (Target Vector Shape):   {y.shape[0]} labels.")
    print("Class Balance:")
    print(f"  Class 1 (Space Debris / Junk): {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"  Class 0 (Payload Satellite):  {(y == 0).sum()} ({(1-y.mean())*100:.2f}%)")

    return X, y, feature_columns


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    WHAT: Split dataset into 80% Training and 20% Unseen Test set, then fit scaler ONLY on train set.
    WHY:  - Unseen test set evaluates how well the model generalizes to new satellite track data.
          - Fitting scaler ONLY on training data prevents Data Leakage (peeking at test statistics).
    HOW:  - train_test_split with stratify=y preserves exact class distribution in both splits.
          - StandardScaler transforms features to mean=0, std=1 for Logistic Regression optimization.
    """
    print("\n--- STEP 4: TRAIN/TEST SPLIT & FEATURE SCALING ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training Set Size:   {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing Set Size:    {X_test.shape[0]} samples ({test_size*100:.0f}%) [Kept Unseen]")

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test data using TRAIN statistics!

    # Return DataFrame wrappers to preserve feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    return X_train, X_test, X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler


def get_preprocessed_pipeline_data(data_path='data/dataset.csv'):
    """
    End-to-end wrapper function for data preparation pipeline.
    """
    df_raw = load_raw_data(data_path)
    df_clean = clean_data(df_raw)
    df_engineered = engineer_features(df_clean)
    X, y, feature_cols = select_features_and_target(df_engineered)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

    return {
        'df_clean': df_clean,
        'X': X,
        'y': y,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


if __name__ == "__main__":
    print("Testing Data Preprocessing Pipeline directly...")
    data_dict = get_preprocessed_pipeline_data()
    print("\n[SUCCESS] Data Preprocessing Pipeline executed cleanly!")
