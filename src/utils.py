"""
Malachi Eberly
utils.py
"""

import logging
import time
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def setup_logger(log_file="logs.txt"):
    """Logger setup"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

class Timer:
    """Timer utility"""
    def __init__(self):
        self.start_time = None

    def start(self):
        """Start timer"""
        self.start_time = time.time()

    def stop(self):
        """Stop timer"""
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

def save_model(model, path="models/icu_los_model.pth"):
    """Save model"""
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def load_model(model_class, path="models/icu_los_model.pth", input_dim=7):
    """Load model"""
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    print("Model loaded successfully")
    return model

def normalize_input(user_input, scaler):
    """Normalize input"""
    user_input = np.array(user_input).reshape(1, -1)
    return scaler.transform(user_input)
