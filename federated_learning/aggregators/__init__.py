"""
Federated Learning Aggregators: FedAvg, FedProx, FedDANE
"""
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np


class BaseFederatedAggregator:
    """Base class for federated aggregators."""

    def aggregate(self, client_models: List[nn.Module], global_model: nn.Module) -> None:
        """
        Aggregate client models into global model.

        Args:
            client_models: List of trained client models
            global_model: Global model to update
        """
        raise NotImplementedError


class FedAvgAggregator(BaseFederatedAggregator):
    """
    Federated Averaging (FedAvg) - Standard federated learning.
    Averages all client model parameters equally.
    """

    def aggregate(self, client_models: List[nn.Module], global_model: nn.Module) -> None:
        """
        Aggregate using FedAvg (simple averaging).

        Args:
            client_models: List of trained client models
            global_model: Global model to update
        """
        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [client.state_dict()[key].float() for client in client_models], 0
            ).mean(0)

        global_model.load_state_dict(global_dict)

    def aggregate_with_weights(
        self, client_models: List[nn.Module], weights: List[float], global_model: nn.Module
    ) -> None:
        """
        Aggregate using weighted averaging (accounts for different dataset sizes).

        Args:
            client_models: List of trained client models
            weights: Relative weights for each client (should sum to 1)
            global_model: Global model to update
        """
        if len(client_models) != len(weights):
            raise ValueError("Number of models must match number of weights")

        if not np.isclose(sum(weights), 1.0):
            weights = np.array(weights) / sum(weights)

        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            weighted_params = [
                w * client.state_dict()[key].float() for w, client in zip(weights, client_models)
            ]
            global_dict[key] = torch.stack(weighted_params, 0).sum(0)

        global_model.load_state_dict(global_dict)


class FedProxAggregator(BaseFederatedAggregator):
    """
    Federated Proximal (FedProx) - Improves convergence under heterogeneity.
    Uses proximal term to prevent client drift.
    Aggregation is same as FedAvg; the difference is in client training.
    """

    def __init__(self, mu: float = 0.1):
        """
        Initialize FedProx aggregator.

        Args:
            mu: Proximal term coefficient (default: 0.1)
        """
        self.mu = mu

    def aggregate(self, client_models: List[nn.Module], global_model: nn.Module) -> None:
        """
        Aggregate using FedProx (same as FedAvg at aggregation step).

        Args:
            client_models: List of trained client models
            global_model: Global model to update
        """
        # FedProx aggregation is identical to FedAvg
        # The proximal term is applied during client training
        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [client.state_dict()[key].float() for client in client_models], 0
            ).mean(0)

        global_model.load_state_dict(global_dict)


class FedDANEAggregator(BaseFederatedAggregator):
    """
    Federated Dual Averaging with NEstorov (FedDANE).
    More robust to non-IID data through variance reduction.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize FedDANE aggregator.

        Args:
            learning_rate: Server-side learning rate
            momentum: Momentum coefficient for dual averaging
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.server_drift = None

    def aggregate(
        self,
        client_models: List[nn.Module],
        global_model: nn.Module,
        client_updates: List[Dict] = None,
    ) -> None:
        """
        Aggregate using FedDANE with variance reduction.

        Args:
            client_models: List of trained client models
            global_model: Global model to update
            client_updates: Optional list of client gradients for variance reduction
        """
        global_dict = global_model.state_dict()
        num_clients = len(client_models)

        # Compute average model
        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = torch.stack(
                [client.state_dict()[key].float() for client in client_models], 0
            ).mean(0)

        # Initialize server drift if not exists
        if self.server_drift is None:
            self.server_drift = {key: torch.zeros_like(v) for key, v in avg_dict.items()}

        # Update with variance reduction
        for key in global_dict.keys():
            # Gradient: difference between average and global model
            gradient = avg_dict[key] - global_dict[key]

            # Update server drift (momentum term)
            self.server_drift[key] = (
                self.momentum * self.server_drift[key] + gradient
            )

            # Update global model with learning rate
            global_dict[key] = global_dict[key] + self.learning_rate * self.server_drift[key]

        global_model.load_state_dict(global_dict)

    def reset_drift(self) -> None:
        """Reset server drift for new training phase."""
        self.server_drift = None
