"""
fusion_gru.py

This module implements the FusionGRU, a variation of the Gated Recurrent Unit (GRU) 
that introduces a fusion mechanism to adaptively combine input differences and 
hidden state updates.

Class:
- FusionGRU(nn.Module):
  - Implements a custom GRU-like structure with a fusion gate.
  - Uses delta-based gating for more adaptive sequence modeling.

Methods (FusionGRU):
- forward(h_it, h_it_m1, hidden_state):
  - Computes the next hidden state using a fusion gate.
  - Introduces delta computation to enhance temporal feature modeling.

- process_sequence(inputs, hidden_state=None):
  - Processes an input sequence step-by-step using the FusionGRU mechanism.
  - Returns the hidden states for all time steps.

"""


import torch
import torch.nn as nn

class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionGRU, self).__init__()
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wg = nn.Linear(input_size * 2, input_size)
        self.bg = nn.Parameter(torch.zeros(input_size))

    def forward(self, h_it, h_it_m1, hidden_state):

        delta_it = h_it - h_it_m1

        concat_input = torch.cat((delta_it, h_it), dim=-1)
        g_t = torch.sigmoid(self.Wg(concat_input) + self.bg)

        x_t = g_t * delta_it + (1 - g_t) * h_it

        combined = torch.cat((x_t, hidden_state), dim=-1)
        z_t = torch.sigmoid(self.Wz(combined))
        r_t = torch.sigmoid(self.Wr(combined))

        r_h_prev = r_t * hidden_state
        combined_candidate = torch.cat((x_t, r_h_prev), dim=-1)
        h_tilde = torch.tanh(self.Wh(combined_candidate))

        h_next = (1 - z_t) * hidden_state + z_t * h_tilde
        return h_next

    def process_sequence(self, inputs, hidden_state=None):
        batch_size, seq_len, input_size = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        h_it_m1 = torch.zeros(batch_size, input_size, device=inputs.device)

        for t in range(seq_len):
            h_it = inputs[:, t, :]  
            hidden_state = self.forward(h_it, h_it_m1, hidden_state)  
            outputs.append(hidden_state.unsqueeze(1))  
            h_it_m1 = h_it  

        return torch.cat(outputs, dim=1)  