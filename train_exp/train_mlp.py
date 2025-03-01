"""
train_mlp.py

This script implements the training process for the SimpleMLP model.
It includes:
- Data loading and preprocessing.
- Model initialization for SimpleMLP.
- Training loop with sequence processing, loss computation, and optimization.
- Hyperparameter configurations such as batch size, learning rate, and hidden sizes.
- Model validation and saving the best model based on performance metrics.
- Evaluation of model performance using metrics like accuracy, precision, recall, F1-score, and AUC.

"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loading.model_loader import load_data_and_model
from embedding.embedding_utils import precompute_all_node_embeddings
from models.MLP import SimpleMLP
from models.MLP import test_model_mlp
import numpy as np
from utils.seed_utils import set_seed
import os

def train_mlp(config):
    seed = config.get("seed", 37719)
    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 2)
    learning_rate = config.get("learning_rate", 3e-4)
    lamb = config.get("lamb", 1)
    csv_path = config.get("csv_path" , None)
    data_path = config.get("data_path" , "CiteSeer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config.get('save_dir' , None)
    mlp_hidden_size1 = config.get("mlp_hidden_size1" , 16)
    mlp_hidden_size2 = config.get("mlp_hidden_size2" , 8)
    mlp_output_size = config.get("mlp_output_size" , 2)



    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_value_list = []

    for seed_now in seed:
        set_seed(seed_now)
        train_loader, val_loader, test_loader, target_model, max_k_core, all_embeddings, dataset, data = load_data_and_model(
            csv_path=csv_path,
            batch_size=batch_size,
            seed=seed_now,
            data_path=data_path,
            lamb = lamb
        )
        input_size = dataset.num_classes
        model = SimpleMLP(input_size, mlp_hidden_size1, mlp_hidden_size2, mlp_output_size).to(device)
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
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
                avg_inputs = all_inputs.mean(dim=1)  

                outputs = model(avg_inputs) 
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                all_labels = []
                all_predictions = []
                for batch_seqs, batch_labels in val_loader:
                    batch_labels = batch_labels.to(device)
                    batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
                    padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
                    max_seq_len = padded_seqs.size(1)
                    all_inputs = []
                    for t in range(max_seq_len):
                        node_indices = padded_seqs[:, t].tolist()
                        cur_inputs = all_embeddings[node_indices]
                        all_inputs.append(cur_inputs)

                    all_inputs = torch.stack(all_inputs, dim=1).to(device)
                    avg_inputs = all_inputs.max(dim=1).values  

                    outputs = model(avg_inputs)
                    _, predicted = torch.max(outputs, 1)

                    all_labels.extend(batch_labels.cpu().tolist())
                    all_predictions.extend(predicted.cpu().tolist())

                accuracy = accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
                recall = recall_score(all_labels, all_predictions, average='binary')
                f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)

                # if recall > best_val_recall:
                #     best_val_recall = recall
                #     torch.save({
                #         'mlp_state_dict': model.state_dict(),
                #         'epoch': epoch + 1,
                #         'best_val_recall': best_val_recall,
                #         'accuracy': accuracy,
                #         'precision': precision,
                #         'f1': f1
                #     }, f"best_mlp_model_lamb_{lamb}_.pth")

        model.eval()
        with torch.no_grad():
            accuracy, precision, recall, f1, auc_value = test_model_mlp(
                model, test_loader, target_model, data, all_embeddings, device
            )
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            auc_value_list.append(auc_value)
        torch.save({
            'model': model.state_dict(),
        }, os.path.join(save_dir, f'{seed_now}.pth'))

    # def print_stats(name, values):
    #     print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    # print_stats("Accuracy", accuracy_list)
    # print_stats("Precision", precision_list)
    # print_stats("Recall", recall_list)
    # print_stats("F1 Score", f1_list)
    # print_stats("Auc", auc_value_list)
    return {
        "accuracy": np.mean(accuracy_list),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1_score": np.mean(f1_list),
        "auc": np.mean(auc_value_list),
        "accuracy_std": np.std(accuracy_list),
        "precision_std": np.std(precision_list),
        "recall_std": np.std(recall_list),
        "f1_score_std": np.std(f1_list),
        "auc_std": np.std(auc_value_list)
    }
