"""
standard_lstm.py

This module defines a standard Long Short-Term Memory (LSTM) network for sequence processing.

Class:
- StandardLSTM(nn.Module):
  - Implements a multi-layer LSTM for processing sequential data.

Methods (StandardLSTM):
- forward(x):
  - Processes an input sequence x through the LSTM.
  - Returns the output sequence along with the final hidden and cell states.

- process_sequence(x):
  - Processes an input sequence and returns only the output sequence.
"""


import torch
import torch.nn as nn

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)

    def process_sequence(self, x):
        outputs, _ = self.forward(x)
        return outputs
