"""
standard_gru.py

This module defines a standard Gated Recurrent Unit (GRU) implementation for sequence processing.

Class:
- StandardGRU(nn.Module):
  - Implements a simple GRU-based recurrent neural network.

Methods (StandardGRU):
- forward(x, hidden_state=None):
  - Takes an input sequence and an optional hidden state.
  - Returns the output sequence and the final hidden state.

- process_sequence(x, hidden_state=None):
  - Processes an input sequence using the GRU and returns the output states.
"""


import torch
import torch.nn as nn

class StandardGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_state=None):
        """
        x: [batch_size, seq_len, input_size]
        """
        output, hidden_n = self.gru(x, hidden_state)
        return output, hidden_n

    def process_sequence(self, x, hidden_state=None):
        output, hidden_n = self.forward(x, hidden_state)
        return output  # [batch_size, seq_len, hidden_size]
