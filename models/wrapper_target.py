"""
wrapper_target.py

This module defines the `TargetGCN` class, which serves as a wrapper around a trained GCN model, 
allowing for efficient querying and embedding extraction.

Features:
- `TargetGCN`: Encapsulates a pre-trained GCN model for inference.
- `predict(query_indices)`: Returns softmax probabilities for the given node indices.
- `get_embedding()`: Extracts the final-layer embeddings of all nodes in the graph.

Usage:
This wrapper is useful for tasks requiring node classification or feature extraction from a 
trained graph convolutional network (GCN).
"""


import torch
import torch.nn.functional as F

class TargetGCN:
    def __init__(self, trained_model, data):
        self.model = trained_model
        self.data = data

    def predict(self, query_indices):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(output[query_indices], dim=1).cpu().numpy()
        return probs

    def get_embedding(self):
        self.model.eval()
        with torch.no_grad():
            _, embeddings = self.model(self.data.x, self.data.edge_index)
        return embeddings
