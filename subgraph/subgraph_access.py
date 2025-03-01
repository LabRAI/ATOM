"""
subgraph_access.py

This script provides functions for processing subgraph data, including:
- Computing PageRank centrality scores for nodes in a subgraph.
- Extracting and returning subgraph features, adjacency matrices, and computed centrality scores.

Functions:
- pagerank_centrality: Computes PageRank centrality for a given adjacency matrix.
- subgraph_access_fn: Returns subgraph features, adjacency matrix, and PageRank centrality scores.
"""

import torch
import networkx as nx

def pagerank_centrality(adj_matrix_coo):
    G = nx.convert_matrix.from_scipy_sparse_array(adj_matrix_coo)
    centrality = nx.pagerank(G)
    normcen = torch.tensor(list(centrality.values()))
    return normcen


def subgraph_access_fn(subgraph_features, subgraph_adj_matrix, num_sample_nodes):
    """
    返回 (features, adj_matrix_coo, pagerank_center)
    """
    features = subgraph_features
    adj_matrix = subgraph_adj_matrix
    normcen = pagerank_centrality(adj_matrix)
    return features, adj_matrix, normcen
