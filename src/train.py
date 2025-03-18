"""
Malachi Eberly
train.py
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.model import ICU_LOS_Model
from src.preprocess import load_data
from src.utils import save_model, Timer

EPOCHS = 1000
LEARNING_RATE = 0.01
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32

class ICUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data
print("Loading data...")
df = load_data()

feature_columns = [
    "age", "gender", "heart_rate", "blood_pressure", 
    "temperature", "spo2", "respiratory_rate"
]
X = df[feature_columns].values
y = df["los"].values

# Train-test split
print("Splitting training and testing data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Convert to ICUDataset
print("Converting to ICUDataset with PyTorch tensors...")
train_dataset = ICUDataset(X_train, y_train)
test_dataset = ICUDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
print("Creating model...")
model = ICU_LOS_Model(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print("Beginning training...")
timer = Timer()
timer.start()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Log training epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
timer.stop()

# Save Model
save_model(model, "models/icu_los_model.pth")
