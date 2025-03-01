"""
age_attack.py

This module implements the AGE (Active Graph Extraction) attack, which strategically queries 
nodes to extract a surrogate model of a target GCN model. The attack selects queries based 
on entropy, diversity, and centrality measures.

Classes:
- AGEAttack(BaseAttack):
  - Implements an active learning-based attack to optimize query selection.
  - Uses entropy and diversity to maximize information gain.
  - Incorporates centrality-based sampling to improve efficiency.
  - Trains an extracted model based on queried responses.

Methods:
- calculate_entropy(predictions): Computes entropy for uncertainty estimation.
- calculate_diversity(predictions): Uses K-Means clustering to estimate diversity.
- generate_queries(features, adj_matrix_coo, normcen, local_to_global): Selects query nodes based on entropy, diversity, and centrality.
- attack(subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10):
  - Iteratively queries nodes, updates the extracted model, and refines predictions.
"""


import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import torch

from .base import BaseAttack
from ..subgraph.subgraph_access import pagerank_centrality

class AGEAttack(BaseAttack):
    def __init__(self, target_model, extracted_model, subgraph_access_fn, query_budget, NCL):
        self.target_model = target_model
        self.extracted_model = extracted_model
        self.subgraph_access_fn = subgraph_access_fn
        self.query_budget = query_budget
        self.NCL = NCL

        self.queries_local = []
        self.queries_global = []
        self.labels = []

    def calculate_entropy(self, predictions):
        return entropy(predictions.T)

    def calculate_diversity(self, predictions):
        num_samples = predictions.shape[0]
        n_clusters = min(self.NCL, num_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(predictions)
        dist = euclidean_distances(predictions, kmeans.cluster_centers_)
        return np.min(dist, axis=1)

    def generate_queries(self, features, adj_matrix_coo, normcen, local_to_global):
        preds = self.target_model.predict(range(features.shape[0]))

        ent_vals = self.calculate_entropy(preds)
        div_vals = self.calculate_diversity(preds)
        centrality_vals = torch.tensor([
            (normcen[:i] < normcen[i]).sum().item() / len(normcen)
            for i in range(len(normcen))
        ]).numpy()

        alpha, beta, gamma = 0.3, 0.3, 0.3

        w_ent = ent_vals / np.max(ent_vals) if np.max(ent_vals) != 0 else ent_vals
        w_div = div_vals / np.max(div_vals) if np.max(div_vals) != 0 else div_vals
        w_cen = centrality_vals / np.max(centrality_vals) if np.max(centrality_vals) != 0 else centrality_vals

        scores = alpha * w_ent + beta * w_div + gamma * w_cen

        for ql in self.queries_local:
            scores[ql] = -1

        query_indices_local = np.argsort(-scores)[:self.query_budget]
        query_indices_global = [local_to_global[idx] for idx in query_indices_local]
        return query_indices_local, query_indices_global

    def attack(self, subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10):
        for round_ in range(epochs):
            num_sample_nodes = subgraph_features.shape[0]
            features, adj_coo, normcen = self.subgraph_access_fn(
                subgraph_features, subgraph_adj_matrix, num_sample_nodes
            )

            q_local, q_global = self.generate_queries(features, adj_coo, normcen, local_to_global)
            preds = self.target_model.predict(q_global)

            self.queries_local.extend(q_local)
            self.queries_global.extend(q_global)
            self.labels.extend(preds)

            train_mask = torch.zeros(self.target_model.data.x.shape[0], dtype=torch.bool)
            train_mask[q_global] = True
            self.target_model.data.train_mask = train_mask

            self.extracted_model.train_model(self.target_model.data, optimizer, criterion, epochs=10)

        print("[AGE Attack] Done.")
