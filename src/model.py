"""
Malachi Eberly
model.py
"""

import torch
import torch.nn as nn

class ICU_LOS_Model(nn.Module):
    """ICU Length of Stay Neural Network"""
    def __init__(self, input_dim, hidden_dim=128, dropout_prob=0.5):
        super(ICU_LOS_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        """Forward for model"""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        return output
