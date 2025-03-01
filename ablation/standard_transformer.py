"""
standard_transformer.py

This module implements a standard Transformer encoder for processing sequential data.
It includes a positional encoding module and a transformer-based sequence encoder.

Classes:
- PositionalEncoding(nn.Module):
  - Implements sinusoidal positional encoding to retain sequence order information.

- StandardTransformer(nn.Module):
  - A Transformer-based sequence encoder.
  - Uses an embedding layer, positional encoding, and multiple transformer encoder layers.

Methods:
- PositionalEncoding.forward(x):
  - Adds positional encoding to input tensor x.

- StandardTransformer.forward(x, src_key_padding_mask=None):
  - Processes an input sequence through the transformer encoder.
  - Returns the encoded output.

- StandardTransformer.process_sequence(x, src_key_padding_mask=None):
  - A wrapper function for forward(), primarily used for sequence processing.
"""


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len, batch_size, d_model = x.size()
        x = x + self.pe[:seq_len]
        return x

class StandardTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead=4, num_layers=2):
        super(StandardTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x) 
        x = x.transpose(0, 1)   
        x = self.pos_embedding(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)
        return output

    def process_sequence(self, x, src_key_padding_mask=None):
        return self.forward(x, src_key_padding_mask=src_key_padding_mask)
