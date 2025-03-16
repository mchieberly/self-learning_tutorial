import logging
import time
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Logger setup
def setup_logger(log_file="logs.txt"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

# Timer utility
class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Model saving and loading
def save_model(model, path="models/icu_los_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def load_model(model_class, path="models/icu_los_model.pth", input_dim=6):
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    print("Model loaded successfully")
    return model

# Input normalization
def normalize_input(user_input, scaler):
    user_input = np.array(user_input).reshape(1, -1)
    return scaler.transform(user_input)
