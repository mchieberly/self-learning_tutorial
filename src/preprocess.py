"""
Malachi Eberly
preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

nan_counter = 0
overflow_counter = 0
over_300_counter = 0
other_counter = 0

def calc_age(row):
    """Compute age at ICU admission safely, adjusting dob if necessary."""

    global nan_counter, overflow_counter, over_300_counter, other_counter

    intime = row["intime"]
    dob = row["dob"]

    # If either date is missing, return NaN
    if pd.isnull(intime) or pd.isnull(dob):
        nan_counter += 1
        return np.nan

    # Convert to python datetime objects to avoid overflow issues
    try:
        dt_intime = intime.to_pydatetime()
        dt_dob = dob.to_pydatetime()
    except Exception:
        nan_counter += 1
        return np.nan

    # If dob is in the future relative to intime, assume it needs to be shifted 100 years back.
    if dt_dob.year > dt_intime.year:
        dt_dob = dt_dob.replace(year=dt_dob.year - 100)

    # Compute age safely
    try:
        age = (dt_intime - dt_dob).days / 365.25
    except OverflowError:
        # In case of an overflow, set age to 90
        overflow_counter += 1
        return 90.0

    # If age is 300 or more, set to 90
    if age >= 300:
        over_300_counter += 1
        return 90.0

    other_counter += 1
    return age

def load_data():
    """Load and preprocess MIMIC-III demo dataset."""

    global nan_counter, overflow_counter, over_300_counter, other_counter

    # Define column data types using nullable Int32 for integer columns
    dtype_dict = {
        "subject_id": "Int32",
        "hadm_id": "Int32",
        "icustay_id": "Int32",
        "los": "float32",
        "gender": "category",
        "itemid": "Int32",
        "valuenum": "float32",
        "dob": "string"
    }

    # Read CSV files
    icustays = pd.read_csv("data/ICUSTAYS.csv", dtype=dtype_dict, parse_dates=["intime"], low_memory=False)
    patients = pd.read_csv("data/PATIENTS.csv", dtype=dtype_dict, low_memory=False)
    vitals = pd.read_csv("data/CHARTEVENTS.csv", dtype=dtype_dict, low_memory=False)

    # Ensure column names remain lowercase
    icustays.columns = icustays.columns.str.lower()
    patients.columns = patients.columns.str.lower()
    vitals.columns = vitals.columns.str.lower()

    # Parse patients' dob to datetime
    patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

    # Fill missing values in integer columns before merging
    icustays.fillna({"icustay_id": -1}, inplace=True)
    patients.fillna({"subject_id": -1}, inplace=True)
    vitals.fillna({"icustay_id": -1, "itemid": -1}, inplace=True)

    # Merge icustays with patients to bring in dob for age calculation
    icustays = icustays.merge(patients[["subject_id", "dob"]], on="subject_id", how="left")

    # Calculate age at ICU admission and enforce float32 type
    icustays["age"] = icustays.apply(calc_age, axis=1).astype("float32")

    # Merge icustays with patients to bring in gender information
    df = icustays.merge(patients[["subject_id", "gender"]], on="subject_id", how="left")

    # Select relevant columns
    df = df[["subject_id", "hadm_id", "icustay_id", "los", "age", "gender"]]

    # Convert categorical gender to numeric
    df["gender"] = df["gender"].map({"M": 0, "F": 1}).astype("Int32")

    # Extract vitals from CHARTEVENTS using a pivot table
    vitals_pivot = vitals.pivot_table(index="icustay_id",
                                      columns="itemid",
                                      values="valuenum",
                                      aggfunc="mean").reset_index()

    vitals_map = {
        220045: "heart_rate",
        223761: "temperature",
        220277: "spo2",
        220210: "respiratory_rate",
        220179: "blood_pressure"
    }
    vitals_pivot.rename(columns=vitals_map, inplace=True)

    # Merge vitals with patient data
    df = df.merge(vitals_pivot, on="icustay_id", how="left")

    # Handle missing values by filling with median for numeric columns
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Normalize continuous features
    scaler = StandardScaler()
    numeric_features = ["age", "heart_rate", "blood_pressure", "temperature", "spo2", "respiratory_rate"]
    existing_cols = [col for col in numeric_features if col in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])

    print(f'nan_counter={nan_counter}, overflow_counter={overflow_counter}, over_300_counter={over_300_counter}, other_counter={other_counter}')

    return df
