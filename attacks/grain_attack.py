"""
grain_attack.py

This module implements the GRAIN (Graph-based Receptive-field And Information-aware Node selection) 
attack for model extraction. It selects query nodes based on their receptive fields and 
pairwise distances to maximize information gain while querying the target model.

Classes:
- GrainAttack (BaseAttack):
  - Implements the GRAIN attack strategy for extracting a target graph neural network (GNN).
  - Selects nodes based on their influence on the graph structure and their distance from others.

Functions:
- get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_matrix2):
  - Computes the receptive field score of a selected node.

- get_current_neighbors_dense(cur_nodes, adj_matrix2):
  - Determines the current set of neighboring nodes for the selected set of queries.

- get_max_nnd_node_dense(idx_used, high_score_nodes, min_distance, adj_matrix2, distance_aax, num_ones, dmax, gamma, num_node):
  - Selects the optimal node to query based on the GRAIN strategy.

Methods (GrainAttack):
- generate_queries(features, adj_matrix, normcen, local_to_global):
  - Generates the next batch of query nodes based on a balance of receptive field coverage and node distances.

- attack(subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10):
  - Executes the attack by querying the target model and updating the extracted model iteratively.

"""


import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from .base import BaseAttack

def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_matrix2):
    dense_row = adj_matrix2[selected_node].toarray()
    rec_vec = ((cur_neighbors + dense_row) != 0).astype(int)
    return weighted_score.dot(rec_vec.ravel())

def get_current_neighbors_dense(cur_nodes, adj_matrix2):
    if len(cur_nodes) == 0:
        return 0
    neighbors = (adj_matrix2[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors

def get_max_nnd_node_dense(idx_used, high_score_nodes, min_distance,
                           adj_matrix2, distance_aax, num_ones, dmax, gamma, num_node):
    max_total_score = 0
    max_node = -1
    cur_neighbors = get_current_neighbors_dense(idx_used, adj_matrix2)

    for node in high_score_nodes:
        rec_field = get_receptive_fields_dense(cur_neighbors, node, num_ones, adj_matrix2)
        node_dist = distance_aax[node, :]
        node_dist = np.where(node_dist < min_distance, node_dist, min_distance)
        node_dist = dmax - node_dist
        dist_score = node_dist.dot(num_ones)

        total_score = rec_field / num_node + gamma * dist_score / num_node
        if total_score > max_total_score:
            max_total_score = total_score
            max_node = node

    return max_node


class GrainAttack(BaseAttack):
    def __init__(self, target_model, extracted_model, subgraph_access_fn, query_budget, num_sample_nodes, gamma):
        self.target_model = target_model
        self.extracted_model = extracted_model
        self.subgraph_access_fn = subgraph_access_fn
        self.query_budget = query_budget
        self.num_sample_nodes = num_sample_nodes
        self.gamma = gamma

        self.queries_local = []
        self.queries_global = []
        self.labels = []

    def generate_queries(self, features, adj_matrix, normcen, local_to_global):
        num_node = features.shape[0]
        dmax = np.ones(num_node)
        num_ones = np.ones(num_node)

        distance_aax = euclidean_distances(features.cpu().numpy(), features.cpu().numpy())
        distance_aax = (distance_aax - distance_aax.min()) / (distance_aax.max() - distance_aax.min())
        adj_matrix2 = adj_matrix.dot(adj_matrix)

        min_distance = np.ones(num_node)
        idx_train = []
        idx_avail = list(range(num_node))
        count = 0

        while count < self.query_budget:
            max_node = get_max_nnd_node_dense(
                idx_train, idx_avail, min_distance,
                adj_matrix2, distance_aax, num_ones, dmax, self.gamma, num_node
            )
            if max_node == -1:
                break

            idx_train.append(max_node)
            idx_avail.remove(max_node)
            dist_n = distance_aax[max_node, :]
            min_distance = np.where(min_distance < dist_n, min_distance, dist_n)
            count += 1

        query_indices_global = [local_to_global[i] for i in idx_train]
        return idx_train, query_indices_global

    def attack(self, subgraph_features, subgraph_adj_matrix, local_to_global, optimizer, criterion, epochs=10):
        for round_ in range(epochs):
            features, adj_coo, normcen = self.subgraph_access_fn(
                subgraph_features, subgraph_adj_matrix, self.num_sample_nodes
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

        print("[Grain Attack] Done.")
