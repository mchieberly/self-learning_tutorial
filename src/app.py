import gradio as gr
import torch
import numpy as np
from src.model import ICU_LOS_Model
from src.preprocess import load_data
from src.utils import load_model, normalize_input
from sklearn.preprocessing import StandardScaler

# Load trained model (using the MLP-based model; input_dim remains 7)
model = load_model(ICU_LOS_Model, "models/icu_los_model.pth", input_dim=7)

# Define feature names:
DATA_FEATURES = ["age", "gender", "heart_rate", "blood_pressure", "temperature", "spo2", "respiratory_rate"]
# Display labels in Gradio:
DISPLAY_FEATURES = [
    "Age",
    "Gender (0=M, 1=F)",
    "Heart Rate",
    "Blood Pressure",
    "Temperature (F)",
    "SpO2",
    "Respiratory Rate"
]

# Load data and fit scaler on the actual dataframe features
df = load_data()
scaler = StandardScaler()
scaler.fit(df[DATA_FEATURES])

def predict_los(*user_inputs):
    """Predict Length of Stay (LOS) based on user input."""
    # user_inputs is a tuple of 7 values in the same order as DATA_FEATURES
    user_array = np.array(user_inputs).reshape(1, -1)
    # Normalize
    user_scaled = normalize_input(user_array, scaler)
    # PyTorch
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
    interface.launch(server_name="0.0.0.0", server_port=7860)
