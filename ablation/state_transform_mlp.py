"""
state_transform_mlp.py

This module implements a Multi-Layer Perceptron (MLP) for transforming probability factors
and integrating them with hidden states in reinforcement learning-based extraction models.

Classes:
- StateTransformMLP(nn.Module):
  - A simple MLP for processing probability factors (prob_factors).
  - Transforms the input through two fully connected layers with ReLU activation.

Methods:
- forward(prob_factor):
  - Processes input probability factors.
  - Applies a two-layer transformation with ReLU activation.
  - Returns the transformed output.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class StateTransformMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim , output_dim):
        super(StateTransformMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, prob_factor):
        x = prob_factor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
