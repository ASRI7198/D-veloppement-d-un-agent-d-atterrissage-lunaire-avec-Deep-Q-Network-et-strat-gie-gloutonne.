import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x) -> torch.Tensor:
        x = torch.tensor(x ,dtype=torch.float)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)




