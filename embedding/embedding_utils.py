"""
embedding_utils.py

This module provides various utility functions for computing node embeddings in a graph 
using a trained Graph Convolutional Network (GCN). It includes methods for retrieving 
node embeddings, performing k-core decomposition, and computing embeddings with neighbor 
pooling and k-core scaling.

Functions:
- get_node_embedding(model, data, node_idx): Retrieves the embedding of a specific node.
- get_one_hop_neighbors(data, node_idx): Returns the one-hop neighbors of a given node.
- average_pooling_with_neighbors(model, data, node_idx): Computes the mean-pooled embedding of a node and its neighbors.
- compute_embedding(model, data, node_idx, lamb=1.0): Computes a node's embedding with k-core scaling.
- k_core_decomposition(graph): Performs k-core decomposition and returns k-core values for nodes.
- average_pooling_with_neighbors_batch(model, data, node_indices): Computes mean embeddings for a batch of nodes and their neighbors.
- compute_embedding_batch(target_model, data, k_core_values_graph, max_k_core, node_indices, lamb=1.0): Computes embeddings for a batch of nodes using k-core scaling.
- simple_embedding_batch(target_model, data, node_indices): Computes simple pooled embeddings for a batch of nodes.
- precompute_all_node_embeddings(target_model, data, k_core_values_graph, max_k_core, lamb=1.0): Precomputes embeddings for all nodes in the graph.
- precompute_simple_embeddings(target_model, data): Precomputes simple pooled embeddings for all nodes.

"""


import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def get_node_embedding(model, data, node_idx):
    embeddings=model.get_embedding()
    return embeddings[node_idx]

def get_one_hop_neighbors(data, node_idx):
    edge_index = data.edge_index
    neighbors = edge_index[1][edge_index[0] == node_idx].tolist()
    return neighbors

def average_pooling_with_neighbors(model, data, node_idx):
    embeddings=model.get_embedding()
    neighbors = get_one_hop_neighbors(data, node_idx)  
    neighbors.append(node_idx)  
    neighbor_embeddings = embeddings[neighbors]  
    pooled_embedding = torch.mean(neighbor_embeddings, dim=0)  
    return pooled_embedding

def compute_embedding(model,data,node_idx,lamb=1.0):

    pooled_embedding=average_pooling_with_neighbors(model, data, node_idx)
    k_core_value = torch.tensor(k_core_values_graph[node_idx], dtype=torch.float32)

    scaled_k_core = torch.log(k_core_value) / torch.log(max_k_core)
    scaling_function = 1 + lamb*(torch.sigmoid(scaled_k_core) - 0.5) * 2


    final_embedding = pooled_embedding * scaling_function

    return final_embedding

def k_core_decomposition(graph):
    k_core_dict = nx.core_number(graph)
    return k_core_dict

def average_pooling_with_neighbors_batch(model, data, node_indices):
    embeddings = model.get_embedding()  
    neighbors = [get_one_hop_neighbors(data, idx) for idx in node_indices]

    node_and_neighbors = [torch.tensor([idx] + list(neighbors[i])) for i, idx in enumerate(node_indices)]

    pooled_embeddings = torch.stack([
        embeddings[node_idx_list].mean(dim=0) for node_idx_list in node_and_neighbors
    ])
    return pooled_embeddings

def compute_embedding_batch(target_model, data, k_core_values_graph, max_k_core, node_indices, lamb=1.0):
    pooled_embeddings = average_pooling_with_neighbors_batch(target_model, data, node_indices)
    k_core_values = torch.tensor([k_core_values_graph[node_idx] for node_idx in node_indices], dtype=torch.float32).to(pooled_embeddings.device)

    max_k_core_tensor = torch.log(max_k_core)
    scaled_k_core = torch.log(k_core_values) / max_k_core_tensor
    scaling_function = 1 + lamb * (torch.sigmoid(scaled_k_core) - 0.5) * 2
    final_embeddings = pooled_embeddings * scaling_function.unsqueeze(-1)
    return final_embeddings

def simple_embedding_batch(target_model, data, node_indices):
    pooled_embeddings = average_pooling_with_neighbors_batch(target_model, data, node_indices)
    return pooled_embeddings

def precompute_all_node_embeddings(
    target_model,
    data,
    k_core_values_graph,
    max_k_core,
    lamb=1.0
):
    all_node_indices = list(range(data.num_nodes))
    all_embeddings = compute_embedding_batch(
        target_model,
        data,
        k_core_values_graph,
        max_k_core,
        all_node_indices,
        lamb=lamb
    )
    return all_embeddings

def precompute_simple_embeddings(target_model, data):
    all_node_indices = list(range(data.num_nodes))
    return simple_embedding_batch(target_model, data, all_node_indices)