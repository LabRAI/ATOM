"""
igp_attack.py

This module implements the IGP (Information Gain Propagation) attack, a model extraction technique 
for Graph Neural Networks (GNNs). The attack selects query nodes based on entropy and 
PageRank centrality to maximize information gain when querying the target model.

Classes:
- IGPAttack (BaseAttack):
  - Implements the IGP attack strategy for extracting a target GNN.
  - Selects nodes by combining PageRank scores and entropy to prioritize high-information queries.

Functions:
- get_entropy_contribute(npy_m1, npy_m2):
  - Computes the entropy difference between two probability distributions.

- calculate_entropy(predictions):
  - Computes the entropy of prediction distributions.

- get_max_info_entropy_node_set(idx_used, subgraph_adj_matrix, high_score_nodes, local_to_global):
  - Selects the best nodes based on entropy contribution to maximize information gain.

Methods (IGPAttack):
- generate_queries(subgraph_features, subgraph_adj_matrix, local_to_global, normcen=None):
  - Generates a set of query nodes using entropy and PageRank-based selection.

- attack(subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10, normcen=None):
  - Executes the attack by iteratively querying the target model and updating the extracted model.
"""


import numpy as np
import copy
from scipy.sparse import csr_matrix
from scipy.stats import entropy
import torch

from .base import BaseAttack
from ..subgraph.subgraph_access import pagerank_centrality

class IGPAttack(BaseAttack):
    def __init__(self, target_model, extracted_model, subgraph_access_fn, query_budget, alpha):
        self.target_model = target_model
        self.extracted_model = extracted_model
        self.subgraph_access_fn = subgraph_access_fn
        self.query_budget = query_budget
        self.alpha = alpha

        self.queries_local = []
        self.queries_global = []
        self.labels = []

        self.num_nodes = self.target_model.data.x.shape[0]
        self.num_labels = self.target_model.data.y.max().item() + 1
        self.num_classes = self.num_labels

        self.label_distributions = np.full(
            (self.num_nodes, self.num_classes),
            1.0 / self.num_classes,
            dtype=np.float32
        )

    @staticmethod
    def get_entropy_contribute(npy_m1, npy_m2):
        entro1 = -np.sum(npy_m1 * np.log2(npy_m1 + 1e-12))
        entro2 = -np.sum(npy_m2 * np.log2(npy_m2 + 1e-12))
        return entro1 - entro2

    def calculate_entropy(self, predictions):
        return entropy(predictions.T)

    def get_max_info_entropy_node_set(self, idx_used, subgraph_adj_matrix, high_score_nodes, local_to_global):
        adj_matrix_csr = csr_matrix(subgraph_adj_matrix)
        labels_ = copy.deepcopy(self.label_distributions)

        def compute_node_score(node):
            node_neighbors = adj_matrix_csr.getrow(node).nonzero()[1]
            global_neighbors = [local_to_global[n] for n in node_neighbors]
            adj_neigh = adj_matrix_csr[node_neighbors, :][:, node_neighbors].toarray()
            aay = np.matmul(adj_neigh, labels_[global_neighbors])

            probs = self.target_model.predict([local_to_global[node]])[0]
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]

            if max_prob > 1e-12:
                new_label_dists = labels_.copy()
                new_label = np.zeros(self.num_labels, dtype=np.float32)
                new_label[max_prob_idx] = 1.0
                new_label_dists[local_to_global[node]] = new_label

                aay_new = np.matmul(adj_neigh, new_label_dists[global_neighbors])
                return max_prob * self.get_entropy_contribute(aay, aay_new)
            else:
                return 0

        max_info_node_set = []
        high_score_local = set(high_score_nodes)
        for _ in range(self.query_budget):
            scores = [compute_node_score(n) for n in high_score_local]
            best_idx = np.argmax(scores)
            best_node = list(high_score_local)[best_idx]
            max_info_node_set.append(best_node)

            labels_[local_to_global[best_node]] = self.target_model.predict([local_to_global[best_node]])[0]
            high_score_local.remove(best_node)

        return max_info_node_set

    def generate_queries(self, subgraph_features, subgraph_adj_matrix, local_to_global, normcen=None):
        num_nodes_local = subgraph_features.shape[0]
        all_nodes_local = np.arange(num_nodes_local)

        candidate_nodes_local = [n for n in all_nodes_local if n not in self.queries_local]
        if len(candidate_nodes_local) <= self.query_budget:
            return np.array(candidate_nodes_local)

        predictions = self.target_model.predict(range(num_nodes_local))
        ent_vals = self.calculate_entropy(predictions)
        pr_scores = pagerank_centrality(subgraph_adj_matrix)

        candidate_scores = []
        for i, node in enumerate(candidate_nodes_local):
            combined_score = self.alpha * pr_scores[node].item() + (1 - self.alpha)*ent_vals[i]
            candidate_scores.append((node, combined_score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        pre_selection_size = min(len(candidate_scores), self.query_budget)
        pre_high_score_nodes = [n for (n, _) in candidate_scores[:pre_selection_size]]

        selected_nodes_local = self.get_max_info_entropy_node_set(
            self.queries_local,
            subgraph_adj_matrix,
            pre_high_score_nodes,
            local_to_global
        )
        return np.array(selected_nodes_local)

    def attack(self, subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10, normcen=None):
        for round_ in range(epochs):
            query_indices_local = self.generate_queries(
                subgraph_features, subgraph_adj_matrix, local_to_global, normcen
            )
            query_indices_global = [local_to_global[i] for i in query_indices_local]
            predictions = self.target_model.predict(query_indices_global)

            self.queries_local.extend(query_indices_local)
            self.queries_global.extend(query_indices_global)
            self.labels.extend(predictions)

            train_mask = torch.zeros(self.target_model.data.x.shape[0], dtype=torch.bool)
            train_mask[query_indices_global] = True
            self.target_model.data.train_mask = train_mask

            self.extracted_model.train_model(self.target_model.data, optimizer, criterion, epochs=10)

        print("[IGP Attack] Done.")
