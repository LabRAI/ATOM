"""
metrics.py

This script defines evaluation metrics for assessing the performance of an extracted model 
compared to a target model in the context of model extraction attacks. It includes:
- Fidelity: Measures how often the extracted model produces the same predictions as the target model.
- Accuracy: Evaluates the extracted model's performance using ground-truth labels.
- Query sequence logging: Outputs the sequence of queried nodes for debugging.
- Cover rate: Computes the percentage of nodes covered in the extracted dataset.

Functions:
- calculate_fidelity(target_model, extracted_model, data)
- calculate_accuracy(extracted_model, data)
- print_query_sequence(queries_global)
- calculate_cover_rate(num_sample_nodes, queried_nodes)
"""


import torch
import torch.nn.functional as F
import numpy as np

def calculate_fidelity(target_model, extracted_model, data):
    target_probs = target_model.predict(range(data.x.shape[0]))
    target_preds = np.argmax(target_probs, axis=1)

    extracted_model.eval()
    with torch.no_grad():
        out, _ = extracted_model(data.x, data.edge_index)
    ext_preds = out.argmax(dim=1).cpu().numpy()

    fidelity = (ext_preds == target_preds).mean()
    return fidelity

def calculate_accuracy(extracted_model, data):
    extracted_model.eval()
    with torch.no_grad():
        out, _ = extracted_model(data.x, data.edge_index)
    preds = out.argmax(dim=1).cpu().numpy()
    true_y = data.y.cpu().numpy()
    acc = (preds == true_y).mean()
    return acc

def print_query_sequence(queries_global):
    print("Query sequence (global index): ", queries_global)

def calculate_cover_rate(num_sample_nodes, queried_nodes):
    return (queried_nodes / num_sample_nodes) * 100
