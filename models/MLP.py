"""
MLP.py

This script defines two Multi-Layer Perceptron (MLP) models and a function for testing 
model performance on a dataset. It includes:

- `StateTransformMLP`: A small MLP that processes probability factors.
- `SimpleMLP`: A feedforward neural network for classification tasks.
- `test_model_mlp`: A function for evaluating a trained MLP model on a test dataset.

Classes:
- `StateTransformMLP(input_dim, output_dim)`: Transforms probability factors using a two-layer MLP.
- `SimpleMLP(input_size, hidden_size1, hidden_size2, output_size)`: Implements a three-layer MLP with softmax output.

Functions:
- `test_model_mlp(model, test_loader, target_model, data, all_embeddings, device)`: 
  Evaluates model performance using accuracy, precision, recall, F1-score, and AUC.

The script supports binary classification and calculates performance metrics, including ROC-AUC.
"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence


class StateTransformMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StateTransformMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, prob_factor):
        x = prob_factor  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.softmax(x)
    
def test_model_mlp(model, test_loader, target_model, data, all_embeddings, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for batch_seqs, batch_labels in test_loader:
            batch_labels = batch_labels.to(device)
            batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
            padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)

            all_inputs = []
            for t in range(padded_seqs.size(1)):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)

            all_inputs = torch.stack(all_inputs, dim=1).to(device)
            avg_inputs = all_inputs.max(dim=1).values
            outputs = model(avg_inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(batch_labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_value = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_value = None
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_value
    }
    return accuracy, precision, recall, f1, auc_value