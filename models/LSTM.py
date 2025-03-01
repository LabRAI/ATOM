"""
LSTM.py

This script defines and trains a Standard LSTM model for sequential data processing 
in conjunction with Proximal Policy Optimization (PPO). It includes:

- A `StandardLSTM` class implementing a basic LSTM architecture.
- A `custom_reward_function` for computing rewards based on classification outcomes.
- `lstm_validate_model`: A function for evaluating model performance on validation data.
- `test_model_lstm`: A function for testing model performance, including AUC and ROC computation.
- Performance evaluation using accuracy, precision, recall, F1-score, and AUC.

Classes:
- StandardLSTM: Implements a single-layer LSTM for processing sequential inputs.

Functions:
- `custom_reward_function(predicted, label, predicted_distribution=None)`: 
  Computes reward based on classification correctness and label distribution.
- `lstm_validate_model(agent, lstm, mlp_transform, val_loader, target_model, data, ...)`: 
  Evaluates the model on validation data and returns key performance metrics.
- `test_model_lstm(agent, lstm, mlp_transform, test_loader, target_model, data, ...)`: 
  Tests the model and computes performance metrics, including AUC and ROC curves.
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


class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
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

def lstm_validate_model(agent, lstm, mlp_transform, val_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    lstm.eval()
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
            hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)

            all_inputs = []
            for t in range(max_seq_len):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)

            all_inputs = torch.stack(all_inputs, dim=1).to(device)
            hidden_states = lstm.process_sequence(all_inputs)
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

def test_model_lstm(agent, lstm, mlp_transform, test_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    lstm.eval()
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
            hidden_states = lstm.process_sequence(all_inputs)
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

    return accuracy, precision, recall, f1, auc_value
