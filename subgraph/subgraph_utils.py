"""
subgraph_utils.py

This script provides utility functions for extracting connected subgraphs from a given graph dataset.
It includes:
- Selection of a connected subgraph using Breadth-First Search (BFS).
- Extraction of subgraph features, adjacency matrices, and node mappings.
- Conversion of extracted subgraphs into COO (Coordinate List) sparse matrix format.

Functions:
- connected_subgraph_access_fn(data, num_sample_nodes):
  Extracts a connected subgraph from the input graph data using BFS.
  Returns subgraph features, adjacency matrix in COO format, subgraph node indices, and a mapping from local to global indices.

"""

import torch
from collections import deque
from scipy.sparse import coo_matrix

def connected_subgraph_access_fn(data, num_sample_nodes):

    num_nodes = data.x.shape[0]
    features = data.x
    adj_matrix = data.edge_index

    start_node = torch.randint(0, num_nodes, (1,)).item()

    visited = set()
    queue = deque([start_node])
    subgraph_nodes_list = []


    while queue and len(subgraph_nodes_list) < num_sample_nodes:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            subgraph_nodes_list.append(node)
            neighbors = adj_matrix[1][adj_matrix[0] == node].tolist()
            queue.extend(neighbors)

    subgraph_nodes = torch.tensor(subgraph_nodes_list)
    global_to_local = {n: idx for idx, n in enumerate(subgraph_nodes.tolist())}
    local_to_global = {idx: n for idx, n in enumerate(subgraph_nodes.tolist())}

    mask = torch.isin(adj_matrix[0], subgraph_nodes) & torch.isin(adj_matrix[1], subgraph_nodes)
    subgraph_edges = adj_matrix[:, mask]

    subgraph_edges_local = torch.stack([
        torch.tensor([global_to_local[n.item()] for n in subgraph_edges[0]]),
        torch.tensor([global_to_local[n.item()] for n in subgraph_edges[1]])
    ])

    subgraph_features = features[subgraph_nodes]

    subgraph_adj_matrix = coo_matrix(
        (
            torch.ones(subgraph_edges_local.shape[1]),
            (subgraph_edges_local[0], subgraph_edges_local[1])
        ),
        shape=(len(subgraph_nodes), len(subgraph_nodes))
    )

    return subgraph_features, subgraph_adj_matrix, subgraph_nodes, local_to_global
