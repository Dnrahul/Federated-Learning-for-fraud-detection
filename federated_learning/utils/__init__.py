"""
Data Preprocessing and Client Dataset Management
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
import os


class DataPreprocessor:
    """Handles data loading, preprocessing, and client data distribution."""

    def __init__(self):
        """Initialize the data preprocessor."""
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None

    def load_and_preprocess_csvs(
        self,
        file_paths: List[str],
        label_column: str = "Is_Fraud",
        drop_columns: List[str] = None,
        stratify_split: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], int]:
        """
        Load and preprocess multiple CSV files for federated learning.

        Args:
            file_paths: List of paths to CSV files (each represents a client)
            label_column: Name of the label column
            drop_columns: Columns to drop
            stratify_split: Whether to stratify train/test split by label
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Tuple of (list of (train_df, test_df) tuples, number of features)
        """
        if drop_columns is None:
            drop_columns = [
                "Transaction_ID",
                "Timestamp",
                "Is_Cross_Border",
                "Customer_Type",
                "Time_Since_Last_Txn",
                "Txn_Region",
                "Merchant_ID",
                "Customer_ID",
                "Risk_Score",
                "Fraud_Type",
            ]

        # Load all dataframes
        dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No valid CSV files found in {file_paths}")

        # Combine for preprocessing
        combined_df = pd.concat(dfs, ignore_index=True)

        # Drop unused columns
        combined_df.drop(
            columns=[c for c in drop_columns if c in combined_df.columns],
            inplace=True,
            errors="ignore",
        )

        # Encode categorical variables
        for col in combined_df.select_dtypes(include="object").columns:
            if col != label_column:
                le = LabelEncoder()
                combined_df[col] = le.fit_transform(combined_df[col])
                self.label_encoders[col] = le

        # Scale numeric features
        num_cols = combined_df.drop(columns=[label_column], errors="ignore").select_dtypes(
            include=["int64", "float64"]
        ).columns
        self.scaler = StandardScaler()
        combined_df[num_cols] = self.scaler.fit_transform(combined_df[num_cols])

        self.feature_names = combined_df.drop(columns=[label_column], errors="ignore").columns

        # Split back into client datasets
        split_sizes = [len(pd.read_csv(fp)) for fp in file_paths if os.path.exists(fp)]
        client_data = []
        start_idx = 0

        for size in split_sizes:
            end_idx = start_idx + size
            client_df = combined_df.iloc[start_idx:end_idx].reset_index(drop=True)

            # Train/test split
            if stratify_split and label_column in client_df.columns:
                train_df, test_df = train_test_split(
                    client_df,
                    test_size=test_size,
                    stratify=client_df[label_column],
                    random_state=random_state,
                )
            else:
                train_df, test_df = train_test_split(
                    client_df, test_size=test_size, random_state=random_state
                )

            client_data.append((train_df.reset_index(drop=True), test_df.reset_index(drop=True)))
            start_idx = end_idx

        input_dim = len(num_cols)
        return client_data, input_dim

    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_column: str = "Is_Fraud",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from DataFrames.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            label_column: Name of label column
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading

        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Prepare training data
        X_train = torch.tensor(
            train_df.drop(columns=[label_column], errors="ignore").values, dtype=torch.float32
        )
        y_train = torch.tensor(
            train_df[label_column].values if label_column in train_df.columns else 0,
            dtype=torch.long,
        )

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Prepare test data
        X_test = torch.tensor(
            test_df.drop(columns=[label_column], errors="ignore").values, dtype=torch.float32
        )
        y_test = torch.tensor(
            test_df[label_column].values if label_column in test_df.columns else 0,
            dtype=torch.long,
        )

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader

    def create_non_iid_data_split(
        self,
        data: pd.DataFrame,
        num_clients: int,
        label_column: str = "Is_Fraud",
        iid_degree: float = 0.1,
        seed: int = 42,
    ) -> List[pd.DataFrame]:
        """
        Create non-IID (heterogeneous) data splits for clients.

        Args:
            data: Full dataset
            num_clients: Number of clients to split data among
            label_column: Label column name
            iid_degree: Degree of non-IID-ness (0=fully non-IID, 1=IID)
            seed: Random seed

        Returns:
            List of DataFrames, one per client
        """
        np.random.seed(seed)

        # Get unique classes
        classes = data[label_column].unique()
        client_data = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_indices = np.where(data[label_column] == cls)[0]
            np.random.shuffle(cls_indices)

            if iid_degree < 1.0:
                # Non-IID: concentrate samples in fewer clients
                num_clients_per_class = max(1, int(num_clients * iid_degree))
                selected_clients = np.random.choice(num_clients, num_clients_per_class, replace=False)

                indices_per_client = np.array_split(cls_indices, num_clients_per_class)
                for client_idx, data_indices in zip(selected_clients, indices_per_client):
                    client_data[client_idx].extend(data_indices)
            else:
                # IID: distribute evenly
                indices_per_client = np.array_split(cls_indices, num_clients)
                for i, data_indices in enumerate(indices_per_client):
                    client_data[i].extend(data_indices)

        # Create client datasets
        client_dfs = []
        for client_indices in client_data:
            client_df = data.iloc[client_indices].reset_index(drop=True)
            if len(client_df) > 0:
                client_dfs.append(client_df)

        return client_dfs

    def simulate_client_dropout(
        self,
        num_clients: int,
        dropout_rate: float = 0.1,
        seed: int = 42,
    ) -> List[bool]:
        """
        Simulate client dropout (clients not participating in round).

        Args:
            num_clients: Total number of clients
            dropout_rate: Fraction of clients to drop
            seed: Random seed

        Returns:
            List of booleans indicating which clients are active
        """
        np.random.seed(seed)
        return np.random.random(num_clients) > dropout_rate
