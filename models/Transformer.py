"""
Transformer.py

This script implements a Transformer-based sequence processing model, replacing traditional 
recurrent architectures such as GRU, LSTM, and RNN. The StandardTransformer model is used 
in reinforcement learning settings, particularly PPO, for learning representations from 
sequential graph data.

Features:
- `StandardTransformer`: A Transformer-based sequence encoder with multi-head attention.
- `PositionalEncoding`: Applies sinusoidal positional embeddings for better sequence modeling.
- `transformer_validate_model`: Evaluates the model on a validation dataset and computes 
  performance metrics.
- `test_model_transformer`: Tests the trained Transformer model, computing accuracy, 
  precision, recall, F1-score, and AUC.
- `custom_reward_function`: Defines a custom reinforcement learning reward function based 
  on prediction correctness and distribution penalties.
"""

import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def custom_reward_function(predicted, label, predicted_distribution=None):

    reward = 0.0

    if predicted_distribution is not None:
        if predicted_distribution > 0.90:
            reward += -8.0

    if predicted == 1 and label == 0:  
        reward+= -22.0
    if predicted == 0 and label == 1:  
        reward+= -18.0
    if predicted == 1 and label == 1:  
        reward+= 16.0
    if predicted == 0 and label == 0:  
        reward+= 16.0

    return reward

def transformer_validate_model(agent, transformer, mlp_transform, val_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    transformer.eval()
    mlp_transform.eval()
    all_true_labels = []
    all_predicted_labels = []

    total_reward = 0.0
    correct_predictions = 0
    correct_detect=0
    total_attack=0
    total_predictions = 0

    with torch.no_grad():  
        for batch_seqs, batch_labels in val_loader:
            batch_labels = batch_labels.to(device)

            batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
            padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
            mask = (padded_seqs != 0).float().to(device)

            max_seq_len = padded_seqs.size(1)
            all_inputs = []
            for t in range(max_seq_len):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)

            all_inputs = torch.stack(all_inputs, dim=1).to(device)
            src_key_padding_mask = (mask == 0)  

            hidden_states = transformer.process_sequence(
                all_inputs,
                src_key_padding_mask=src_key_padding_mask
            )  
            masked_hidden_states = hidden_states * mask.unsqueeze(-1)

            prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)
            custom_states = (mlp_transform(prob_factors) * masked_hidden_states).detach()

            actions, _, _, _ = agent.select_action(custom_states.view(-1, hidden_size))
            actions = actions.view(len(batch_seqs), max_seq_len)

            for i in range(len(batch_seqs)):
                last_valid_step = (mask[i].sum().long() - 1).item()
                predicted_action = actions[i, last_valid_step].item()
                true_label = batch_labels[i].item()

                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_action)

                if predicted_action == true_label:
                    correct_predictions += 1
                    if true_label==1:
                        correct_detect+=1
                if true_label==1:
                  total_attack+=1
                total_predictions += 1

                reward = custom_reward_function(predicted_action, true_label)
                total_reward += reward

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, average='binary')
    recall = recall_score(all_true_labels, all_predicted_labels, average='binary')
    f1 = f1_score(all_true_labels, all_predicted_labels, average='binary')

    return total_reward, accuracy, precision, recall, f1

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def test_model_transformer(agent, transformer, mlp_transform, test_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    transformer.eval()
    mlp_transform.eval()

    total_reward = 0.0
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []  

    with torch.no_grad(): 
        for batch_seqs, batch_labels in test_loader:
            batch_labels = batch_labels.to(device)

            batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
            padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
            mask = (padded_seqs != 0).float().to(device)

            max_seq_len = padded_seqs.size(1)
            hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)

            all_inputs = []
            for t in range(max_seq_len):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)

            all_inputs = torch.stack(all_inputs, dim=1).to(device)

            src_key_padding_mask = (mask == 0)  

            hidden_states = transformer.process_sequence(
                all_inputs,
                src_key_padding_mask=src_key_padding_mask
            )  
            masked_hidden_states = hidden_states * mask.unsqueeze(-1)

            prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)
            custom_states = (mlp_transform(prob_factors) * masked_hidden_states).detach()

            actions, probabilities, _, _ = agent.select_action(custom_states.view(-1, hidden_size))
            actions = actions.view(len(batch_seqs), max_seq_len)
            probabilities = probabilities.view(len(batch_seqs), max_seq_len)  

            for i in range(len(batch_seqs)):
                last_valid_step = (mask[i].sum().long() - 1).item()
                predicted_action = actions[i, last_valid_step].item()
                predicted_prob = probabilities[i, last_valid_step].item()  
                true_label = batch_labels[i].item()

                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_action)
                all_predicted_probs.append(predicted_prob)  

                reward = custom_reward_function(predicted_action, true_label)
                total_reward += reward

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, average='binary')
    recall = recall_score(all_true_labels, all_predicted_labels, average='binary')
    f1 = f1_score(all_true_labels, all_predicted_labels, average='binary')

    try:
        fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_probs)
        auc_value = auc(fpr, tpr)
    except ValueError:
        print("Unable to compute AUC due to only one class in true labels.")

    print(f"Test Results - Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}, "
          f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}, Auc {auc_value:.2%}")
    return accuracy, precision, recall, f1, auc_value



class StandardTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead=4, num_layers=2):
        super(StandardTransformer, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)

        self.pos_embedding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,  
            batch_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)

        x = x.transpose(0, 1)  
        x = self.pos_embedding(x)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        output = output.transpose(0, 1)
        return output

    def process_sequence(self, inputs, src_key_padding_mask=None):
        return self.forward(inputs, src_key_padding_mask=src_key_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x