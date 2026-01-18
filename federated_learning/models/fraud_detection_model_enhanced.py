"""
Enhanced Fraud Detection Model with Attention Mechanism and Batch Normalization
"""
import torch
import torch.nn as nn


class FraudDetectionModelEnhanced(nn.Module):
    """
    Enhanced neural network with attention and batch normalization for fraud detection.
    Features: Batch Normalization, Attention Weights, Better regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        hidden_dim3: int = 32,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize enhanced fraud detection model.

        Args:
            input_dim: Dimension of input features
            hidden_dim1: Size of first hidden layer (default: 128)
            hidden_dim2: Size of second hidden layer (default: 64)
            hidden_dim3: Size of third hidden layer (default: 32)
            dropout_rate: Dropout rate for regularization
        """
        super(FraudDetectionModelEnhanced, self).__init__()

        self.input_dim = input_dim

        # Feature transformation with batch normalization
        self.bn_input = nn.BatchNorm1d(input_dim)

        # Main network
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc_out = nn.Linear(hidden_dim3, 2)

        # Attention layer
        self.attention = nn.Linear(hidden_dim3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        # Normalize input
        x = self.bn_input(x)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        attention_weights = torch.sigmoid(self.attention(x))
        x = x * attention_weights  # Apply attention

        x = self.dropout3(x)

        # Output
        logits = self.fc_out(x)
        return logits

    def get_model_size(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
