"""
Malachi Eberly
app.py
"""

import gradio as gr
import torch
import numpy as np
from src.model import ICU_LOS_Model
from src.preprocess import load_data
from src.utils import load_model, normalize_input
from sklearn.preprocessing import StandardScaler

SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860

# Load trained model
model = load_model(ICU_LOS_Model, "models/icu_los_model.pth")

# Define feature names and display labels for Gradio
DATA_FEATURES = ["age", "gender", "heart_rate", "blood_pressure", "temperature", "spo2", "respiratory_rate"]
DISPLAY_FEATURES = [
    "Age (years)",
    "Gender (0=M, 1=F)",
    "Heart Rate (bpm)",
    "Blood Pressure (mmHg)",
    "Temperature (°F)",
    "SpO2 (%)",
    "Respiratory Rate (breaths/min)"
]

# Load data and fit scaler
df = load_data()
scaler = StandardScaler()
scaler.fit(df[DATA_FEATURES])

def predict_los(*user_inputs):
    """Predict Length of Stay (LOS) based on user input."""
    # user_inputs is a tuple of 7 values in the same order as DATA_FEATURES
    user_array = np.array(user_inputs).reshape(1, -1)
    # Normalize
    user_scaled = normalize_input(user_array, scaler)
    input_tensor = torch.tensor(user_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return round(prediction, 2)

# Create Gradio inputs with display labels
inputs = [gr.Number(label=label) for label in DISPLAY_FEATURES]

interface = gr.Interface(
    fn=predict_los,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Length of Stay (days)"),
    title="ICU Length of Stay Prediction",
    description="Enter patient details to predict ICU length of stay. Note that temperature is in Fahrenheit by default.",
)

if __name__ == "__main__":
    interface.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
