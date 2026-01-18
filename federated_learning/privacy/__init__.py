"""
Privacy-Preserving Federated Learning Module
Includes Differential Privacy (DP-SGD), Privacy Accounting, and Membership Inference Attack
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from scipy import special


class DifferentialPrivacyEngine:
    """
    Implements Differential Privacy for Federated Learning using DP-SGD.
    Supports both client-level and sample-level DP.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        target_epsilon: float = None,
        dp_type: str = "sample",
    ):
        """
        Initialize Differential Privacy Engine.

        Args:
            noise_multiplier: Ratio of noise std to gradient clipping norm
            max_grad_norm: Maximum allowed L2 norm of gradients (clipping bound)
            delta: Delta value for (epsilon, delta)-DP
            target_epsilon: Target epsilon value (if known in advance)
            dp_type: 'sample' for sample-level DP, 'client' for client-level DP
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.target_epsilon = target_epsilon
        self.dp_type = dp_type
        self.privacy_spent = []  # Track (epsilon, delta) over rounds

    def clip_gradients(self, model: nn.Module, max_norm: float = None) -> float:
        """
        Clip gradients of model parameters to max_norm.

        Args:
            model: PyTorch model
            max_norm: Maximum norm for gradient clipping (uses self.max_grad_norm if None)

        Returns:
            Actual norm of gradients before clipping
        """
        if max_norm is None:
            max_norm = self.max_grad_norm

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2.0)

        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        return total_norm

    def add_gaussian_noise(self, model: nn.Module) -> None:
        """
        Add Gaussian noise to model parameters for differential privacy.

        Args:
            model: PyTorch model to add noise to
        """
        noise_std = self.noise_multiplier * self.max_grad_norm

        for p in model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)

    def add_noise_to_updates(self, updates: List[Dict]) -> List[Dict]:
        """
        Add noise to federated learning updates.

        Args:
            updates: List of model state dicts

        Returns:
            Noisy updates
        """
        noisy_updates = []
        for update in updates:
            noisy_update = {}
            for key, value in update.items():
                if value.dtype.is_floating_point:
                    noise = torch.randn_like(value) * (self.noise_multiplier * self.max_grad_norm)
                    noisy_update[key] = value + noise
                else:
                    noisy_update[key] = value
            noisy_updates.append(noisy_update)

        return noisy_updates

    def compute_privacy_loss_rdp(
        self, num_samples: int, batch_size: int, rounds: int
    ) -> Tuple[float, float]:
        """
        Compute privacy loss using Rényi Differential Privacy (RDP).

        Args:
            num_samples: Total number of training samples
            batch_size: Batch size for training
            rounds: Number of training rounds

        Returns:
            (epsilon, delta) tuple
        """
        q = batch_size / num_samples  # Sampling probability
        noise_multiplier = self.noise_multiplier

        # RDP orders to check
        orders = np.linspace(1.5, 100, 20)
        max_order_epsilon = np.inf
        best_order = None

        for order in orders:
            if order == 1:
                continue
            order_eps = (2 * np.log(1 / self.delta)) / ((order - 1) * q * q)
            order_eps = order_eps / (noise_multiplier**2)
            order_eps = order_eps + np.log(1 + q * noise_multiplier**2 / 2)

            if order_eps < max_order_epsilon:
                max_order_epsilon = order_eps
                best_order = order

        epsilon = max_order_epsilon * np.sqrt(rounds)
        self.privacy_spent.append((epsilon, self.delta))

        return epsilon, self.delta

    def compute_privacy_loss_basic(
        self, num_samples: int, batch_size: int, rounds: int
    ) -> Tuple[float, float]:
        """
        Compute privacy loss using basic composition (more conservative).

        Args:
            num_samples: Total number of training samples
            batch_size: Batch size for training
            rounds: Number of training rounds

        Returns:
            (epsilon, delta) tuple
        """
        q = min(1, batch_size / num_samples)
        # Using basic composition: eps = sqrt(rounds) * log(1 + q^2 / noise_multiplier^2)
        epsilon = np.sqrt(rounds) * np.log(1 + (q**2) / (self.noise_multiplier**2))

        self.privacy_spent.append((epsilon, self.delta))
        return epsilon, self.delta


class MembershipInferenceAttack:
    """
    Membership Inference Attack to estimate privacy leakage.
    Attempts to infer whether a data point was used in training.
    """

    @staticmethod
    def attack_via_loss(
        model: nn.Module,
        train_loader,
        test_loader,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Perform membership inference attack based on model loss.

        Args:
            model: Trained model
            train_loader: DataLoader for training set
            test_loader: DataLoader for test set
            device: Device to run on

        Returns:
            Dictionary with attack metrics (advantage, AUC, precision, recall)
        """
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")

        # Compute loss for training set members
        train_losses = []
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                losses = criterion(logits, y)
                train_losses.extend(losses.cpu().numpy())

        # Compute loss for test set non-members
        test_losses = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                losses = criterion(logits, y)
                test_losses.extend(losses.cpu().numpy())

        # Attack: higher loss = likely non-member, lower loss = likely member
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        threshold = (train_losses.mean() + test_losses.mean()) / 2

        # Compute metrics
        member_correct = np.sum(train_losses <= threshold)
        non_member_correct = np.sum(test_losses > threshold)

        accuracy = (member_correct + non_member_correct) / (len(train_losses) + len(test_losses))
        precision = member_correct / (member_correct + len(test_losses) - non_member_correct + 1e-6)
        recall = member_correct / len(train_losses)
        advantage = accuracy - 0.5

        return {
            "advantage": advantage,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "train_loss_mean": train_losses.mean(),
            "test_loss_mean": test_losses.mean(),
        }
