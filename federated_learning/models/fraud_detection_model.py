"""
Base Fraud Detection Model using PyTorch
"""
import torch
import torch.nn as nn


class FraudDetectionModel(nn.Module):
    """
    Lightweight neural network for binary fraud classification.
    Architecture: Input -> 64 -> ReLU -> 32 -> ReLU -> 2 (class logits)
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32):
        """
        Initialize the fraud detection model.

        Args:
            input_dim: Dimension of input features
            hidden_dim1: Size of first hidden layer (default: 64)
            hidden_dim2: Size of second hidden layer (default: 32)
        """
        super(FraudDetectionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, 2),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        return self.model(x)

    def get_model_size(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
