"""
Phase C: Centralized & Federated Splitting.
Performs stratified centralized Train/Test splitting and Dirichlet (alpha=0.5) non-IID
federated client partitioning for heterogeneous UAV swarms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from .config import PipelineConfig

class FederatedSplitter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def split_centralized(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into stratified Train and Test subsets (80/20).
        """
        feature_cols = [c for c in df.columns if c not in ['canonical_label', 'class_id', 'binary_label']]
        
        # Ensure class stratifiability (fallback to random if min class count < 2)
        class_counts = df['class_id'].value_counts()
        stratify_col = df['class_id'] if class_counts.min() >= 2 else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_col
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def partition_dirichlet(self, train_df: pd.DataFrame, num_clients: int, alpha: float) -> List[pd.DataFrame]:
        """
        Partitions training data among N federated clients using Dirichlet distribution (alpha).
        Simulates non-IID class heterogeneity across UAV swarm nodes.
        """
        np.random.seed(self.config.random_state)
        n_samples = len(train_df)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        classes = train_df['class_id'].unique()
        
        for c in classes:
            idx_c = np.where(train_df['class_id'].values == c)[0]
            np.random.shuffle(idx_c)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance small allocations
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            
            # Split class indices across clients
            splits = np.split(idx_c, proportions)
            for client_id, split_idx in enumerate(splits):
                client_indices[client_id].extend(split_idx)

        client_dfs = []
        for client_id, indices in enumerate(client_indices):
            c_df = train_df.iloc[indices].sample(frac=1.0, random_state=self.config.random_state).reset_index(drop=True)
            client_dfs.append(c_df)
            print(f"    -> Client {client_id+1}: {len(c_df):,} samples | Class breakdown: {c_df['class_id'].nunique()} classes")

        return client_dfs

    def run_splitting(self, harmonized_branches: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Executes Phase C for both Network and Physical branches.
        """
        print("\n========================================================")
        print("  PHASE C: CENTRALIZED & FEDERATED DIRICHLET PARTITIONING")
        print("========================================================")
        split_results = {}

        for branch, df in harmonized_branches.items():
            print(f"\n[*] Partitioning {branch} Branch (Total: {len(df):,} samples)...")
            
            # 1. Centralized Train/Test split
            train_df, test_df = self.split_centralized(df)
            print(f"    Centralized Train: {len(train_df):,} | Test: {len(test_df):,}")

            # 2. Federated Dirichlet Partitioning on Train Split
            print(f"    Federated Dirichlet Partitioning (alpha={self.config.dirichlet_alpha}, clients={self.config.num_federated_clients})...")
            client_dfs = self.partition_dirichlet(
                train_df,
                num_clients=self.config.num_federated_clients,
                alpha=self.config.dirichlet_alpha
            )

            split_results[branch] = {
                "train_df": train_df,
                "test_df": test_df,
                "client_dfs": client_dfs
            }

        return split_results
