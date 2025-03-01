"""
target_model.py

This script defines a Graph Convolutional Network (GCN) model for use as a target model in 
graph-based learning tasks. It includes:

- `GCN`: A two-layer Graph Convolutional Network (GCN) for node classification.
- `train_gcn`: A function for training the GCN model using cross-entropy loss and an optimizer.

Classes:
- `GCN(input_dim, hidden_dim, output_dim)`: Implements a standard two-layer GCN.
- `train_gcn(model, data, optimizer, criterion, epochs=200, verbose=True)`: Trains the GCN 
  model on a given dataset.

The script is designed for use in semi-supervised node classification tasks with 
graph-structured data.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        x = F.relu(hidden)
        output = self.conv2(x, edge_index)
        return F.log_softmax(output, dim=1), output


def train_gcn(model, data, optimizer, criterion, epochs=200, verbose=True):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"[GCN-Train] Epoch {epoch}, Loss: {loss.item()}")
