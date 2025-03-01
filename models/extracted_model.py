"""
extracted_model.py

This script defines the `ExtractedGCN` model, a simple Graph Convolutional Network (GCN) 
used for approximating the outputs of a target model.

Key Features:
- Implements a two-layer GCN with ReLU activation.
- Supports training with a provided dataset, optimizer, and loss function.
- Uses `log_softmax` as the output activation function for classification tasks.
- Includes a training loop with optional verbosity for monitoring progress.

Classes:
- ExtractedGCN: A standard GCN model with a training function.

Methods:
- `forward(x, edge_index)`: 
  Defines the forward pass of the GCN, processing node features and edge indices.
- `train_model(data, optimizer, criterion, epochs=50, verbose=False)`: 
  Trains the GCN model using the specified loss function and optimizer.
"""


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ExtractedGCN(torch.nn.Module):
    """
    与 GCN 相同结构，用来拟合目标模型的输出。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExtractedGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        x = F.relu(hidden)
        output = self.conv2(x, edge_index)
        return F.log_softmax(output, dim=1), output

    def train_model(self, data, optimizer, criterion, epochs=50, verbose=False):
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self(data.x, data.edge_index)
            loss = criterion(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"[Extracted-Train] Epoch {epoch}, Loss: {loss.item()}")
