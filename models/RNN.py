"""
RNN.py

This script defines an RNN-based model (`StandardRNN`) along with functions for validating 
and testing the model using Proximal Policy Optimization (PPO). It includes:

- `StandardRNN`: A simple recurrent neural network with a single-layer RNN.
- `custom_reward_function`: Computes a reward based on model predictions and label distributions.
- `rnn_validate_model`: Evaluates the model on a validation dataset using accuracy, precision, recall, F1-score, and total reward.
- `test_model_rnn`: Tests the trained model on a test dataset and computes performance metrics, including ROC-AUC.

Classes:
- `StandardRNN(input_size, hidden_size, num_layers=1, batch_first=True)`: Implements an RNN-based sequence model.
- `rnn_validate_model(agent, rnn, mlp_transform, val_loader, ...)`: Performs model validation.
- `test_model_rnn(agent, rnn, mlp_transform, test_loader, ...)`: Tests model performance and calculates AUC.

The script integrates a PPO-based action-selection mechanism for sequence classification.

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

class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(StandardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x):
        outputs, h_n = self.rnn(x)  
        return outputs, h_n

    def process_sequence(self, x):
        outputs, _ = self.forward(x)
        return outputs
def rnn_validate_model(agent, rnn, mlp_transform, val_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    rnn.eval()
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
            hidden_states = rnn.process_sequence(all_inputs)
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

def test_model_rnn(agent, rnn, mlp_transform, test_loader, target_model, data, hidden_size, all_embeddings, action_dim, device):
    agent.eval()  
    rnn.eval()
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
            hidden_states = rnn.process_sequence(all_inputs)
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
    fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_probs)
    auc_value = auc(fpr, tpr)
    return accuracy, precision, recall, f1, auc_value

