"""
ppo_eval.py

This script contains evaluation functions for Proximal Policy Optimization (PPO) agents 
in a reinforcement learning setting for graph neural network (GNN) extraction tasks. 
It provides validation (`validate_model`) and testing (`test_model`) functions 
to assess model performance using various metrics.

Functions:
- custom_reward_function(predicted, label): Computes a custom reward function for RL training.
- validate_model(agent, gru, mlp_transform, val_loader, target_model, data, all_embeddings, hidden_size, action_dim, device):
  Evaluates the agent on the validation set, computing accuracy, precision, recall, and F1-score.
- test_model(agent, gru, mlp_transform, test_loader, target_model, data, all_embeddings, hidden_size, device):
  Tests the trained agent and generates evaluation metrics, including AUC-ROC.

Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)

def custom_reward_function(predicted, label):
    if predicted == 1 and label == 0:  
        return -22.0  
    if predicted == 0 and label == 1:  
        return -18.0
    if predicted == 1 and label == 1:  
        return 16.0  
    if predicted == 0 and label == 0:  
        return 16.0  

def validate_model(agent, gru, mlp_transform, val_loader, target_model, data, all_embeddings, hidden_size, action_dim, device):
    agent.eval()  
    gru.eval()
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
            hidden_states = gru.process_sequence(all_inputs)
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
    precision = precision_score(all_true_labels, all_predicted_labels, average='binary',zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='binary',zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='binary',zero_division=0)

    return total_reward, accuracy, precision, recall, f1

def test_model(agent, gru, mlp_transform, test_loader, target_model, data, all_embeddings, hidden_size, device):
    agent.eval()  
    gru.eval()
    mlp_transform.eval()

    total_reward = 0.0
    action_dim = 2
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
            hidden_states = gru.process_sequence(all_inputs)
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
