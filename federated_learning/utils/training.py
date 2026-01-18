"""
Training and evaluation utilities for federated learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


class ClientTrainer:
    """Handles client-side training in federated learning."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """
        Initialize client trainer.

        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train_one_round(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        global_model: nn.Module = None,
        mu: float = 0.0,
        use_dp: bool = False,
        dp_engine=None,
    ) -> Dict[str, float]:
        """
        Train client model for one communication round.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of local epochs
            global_model: Global model (for FedProx regularization)
            mu: Proximal term coefficient for FedProx
            use_dp: Whether to use differential privacy
            dp_engine: Differential privacy engine instance

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Compute class weights for imbalanced data
        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.numpy())

        fraud_count = sum(1 for l in all_labels if l == 1)
        nonfraud_count = sum(1 for l in all_labels if l == 0)
        total = fraud_count + nonfraud_count

        if fraud_count > 0:
            weight = torch.tensor(
                [1.0, total / (fraud_count + 1e-6)], dtype=torch.float32
            ).to(self.device)
        else:
            weight = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weight)

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                output = self.model(X)
                loss = criterion(output, y)

                # FedProx regularization
                if global_model is not None and mu > 0:
                    prox_term = sum(
                        ((p - p_global) ** 2).sum()
                        for p, p_global in zip(
                            self.model.parameters(), global_model.parameters()
                        )
                    )
                    loss += (mu / 2) * prox_term

                # Backward pass
                loss.backward()

                # Apply differential privacy
                if use_dp and dp_engine is not None:
                    dp_engine.clip_gradients(self.model)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Add noise after optimization step for DP
                if use_dp and dp_engine is not None:
                    dp_engine.add_gaussian_noise(self.model)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return {"loss": avg_loss, "epochs": epochs}


class ModelEvaluator:
    """Handles model evaluation and metrics computation."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize model evaluator.

        Args:
            device: Device to evaluate on
        """
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        label: str = "Evaluation",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            model: PyTorch model
            data_loader: DataLoader for test data
            label: Label for output printing

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                logits = model(X)
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        confusion = confusion_matrix(all_labels, all_preds)

        # AUC only if we have both classes
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0

        report = classification_report(all_labels, all_preds, output_dict=True)

        print(f"\n📊 Evaluation on {label}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{classification_report(all_labels, all_preds)}")

        return {
            "accuracy": accuracy,
            "auc": auc,
            "confusion_matrix": confusion,
            "report": report,
            "predictions": all_preds,
            "probabilities": all_probs,
            "labels": all_labels,
        }

    @staticmethod
    def plot_confusion_matrix(
        confusion: np.ndarray,
        title: str = "Confusion Matrix",
        figsize: Tuple = (8, 6),
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            confusion: Confusion matrix array
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    @staticmethod
    def plot_roc_pr_curves(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        title: str = "ROC and PR Curves",
        figsize: Tuple = (14, 5),
    ) -> None:
        """
        Plot ROC and Precision-Recall curves.

        Args:
            y_true: True labels
            y_probs: Predicted probabilities
            title: Plot title
            figsize: Figure size
        """
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)

        plt.figure(figsize=figsize)

        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label="PR Curve", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


class TrainingMetricsTracker:
    """Track and visualize training metrics across federated rounds."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            "round": [],
            "global_accuracy": [],
            "client_accuracies": [],
            "global_loss": [],
            "privacy_epsilon": [],
            "privacy_delta": [],
        }

    def log_round(
        self,
        round_num: int,
        global_accuracy: float,
        client_accuracies: list,
        global_loss: float = None,
        privacy_budget: Tuple[float, float] = None,
    ) -> None:
        """
        Log metrics for a training round.

        Args:
            round_num: Round number
            global_accuracy: Global model accuracy
            client_accuracies: List of client accuracies
            global_loss: Global model loss
            privacy_budget: (epsilon, delta) tuple
        """
        self.metrics["round"].append(round_num)
        self.metrics["global_accuracy"].append(global_accuracy)
        self.metrics["client_accuracies"].append(client_accuracies)
        if global_loss is not None:
            self.metrics["global_loss"].append(global_loss)
        if privacy_budget is not None:
            self.metrics["privacy_epsilon"].append(privacy_budget[0])
            self.metrics["privacy_delta"].append(privacy_budget[1])

    def plot_convergence(self, figsize: Tuple = (12, 4)) -> None:
        """Plot training convergence curves."""
        plt.figure(figsize=figsize)

        # Global accuracy
        if self.metrics["global_accuracy"]:
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics["round"], self.metrics["global_accuracy"], marker="o")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Global Model Accuracy")
            plt.grid(True, alpha=0.3)

        # Client accuracies
        if self.metrics["client_accuracies"]:
            plt.subplot(1, 2, 2)
            client_accs = np.array(self.metrics["client_accuracies"])
            for i in range(client_accs.shape[1]):
                plt.plot(self.metrics["round"], client_accs[:, i], marker="o", label=f"Client {i+1}")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Client Accuracies per Round")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_privacy_utility_tradeoff(self, figsize: Tuple = (10, 6)) -> None:
        """Plot privacy-utility trade-off curve."""
        if self.metrics["privacy_epsilon"] and self.metrics["global_accuracy"]:
            plt.figure(figsize=figsize)
            plt.plot(
                self.metrics["privacy_epsilon"],
                self.metrics["global_accuracy"],
                marker="o",
                linewidth=2,
                markersize=8,
            )
            plt.xlabel("Privacy Budget (ε)")
            plt.ylabel("Global Model Accuracy")
            plt.title("Privacy-Utility Trade-off")
            plt.grid(True, alpha=0.3)
            plt.show()
