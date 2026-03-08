import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
