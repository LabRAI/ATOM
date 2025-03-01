"""
model_loader.py

This module provides functions to load dataset-specific graph data, preprocess it, 
initialize a target GCN model, perform k-core decomposition, and precompute node embeddings.

Functions:
- load_data_and_model(csv_path, batch_size, seed, data_path, lamb):
  - Loads data from CSV and initializes PyTorch DataLoaders.
  - Loads graph datasets (CiteSeer, PubMed, Cora, Cora_ML, Cornell, Wisconsin) using PyTorch Geometric.
  - Initializes a GCN-based target model for node classification.
  - Performs k-core decomposition to analyze graph structure.
  - Precomputes node embeddings incorporating k-core values.

Dependencies:
- PyTorch, PyTorch Geometric, NetworkX
- data_loading.build_dataloader (for dataset loading)
- models.wrapper_target (for encapsulating the target GCN model)
- models.target_model (for defining the GCN model)
- embedding.embedding_utils (for embedding computation and k-core decomposition)
"""
import os
from pathlib import Path
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, CitationFull, WebKB
from torch_geometric.utils import to_networkx
from data_loading.build_dataloader import build_loaders
from models.wrapper_target import TargetGCN
from models.target_model import GCN
from embedding.embedding_utils import k_core_decomposition, precompute_all_node_embeddings


def load_data_and_model(csv_path, batch_size, seed, data_path, lamb):
    try:
        script_dir = Path(__file__).resolve().parent
        parent_dir = script_dir.parent
    except NameError:
        parent_dir = Path.cwd().parent
        print("If __file__ is not defined, the directory above the current working directory is used as the target directory.")
    
    os.chdir(parent_dir)

    train_loader, val_loader, test_loader = build_loaders(
        csv_path=csv_path,
        batch_size=batch_size,
        drop_last=True,
        seed=seed
    )

    # ======== Step 2: target_model, data =========
    if data_path == "CiteSeer":
        dataset = Planetoid(root="./data", name=data_path)
        data = dataset[0]
    elif data_path == "PubMed":
        dataset = Planetoid(root="./data", name="PubMed")
        data = dataset[0]
    elif data_path == "Cora":
        dataset = Planetoid(root="./data", name=data_path)
        data = dataset[0]
    elif data_path == "Cora_ML":
        dataset = CitationFull(root="./data", name="Cora_ML")
        data = dataset[0]
        num_nodes = data.num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        num_test = num_nodes - num_train - num_val
        perm = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:num_train]] = True
        data.val_mask[perm[num_train:num_train + num_val]] = True
        data.test_mask[perm[num_train + num_val:]] = True
    elif data_path == "Cornell" or data_path == "Wisconsin":
        dataset = WebKB(root="./data", name=data_path)
        data = dataset[0]
        num_nodes = data.num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        num_test = num_nodes - num_train - num_val

        perm = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:num_train]] = True
        data.val_mask[perm[num_train:num_train + num_val]] = True
        data.test_mask[perm[num_train + num_val:]] = True

    trained_gcn = GCN(dataset.num_features, 16, dataset.num_classes)
    target_model = TargetGCN(trained_model=trained_gcn, data=data)

    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    k_core_values_graph = k_core_decomposition(G)
    max_k_core = torch.tensor(max(k_core_values_graph.values()), dtype=torch.float32)

    all_embeddings = precompute_all_node_embeddings(
        target_model, data, k_core_values_graph, max_k_core, lamb=lamb
    )

    return train_loader, val_loader, test_loader, target_model, max_k_core, all_embeddings, dataset, data
