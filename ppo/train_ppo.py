"""
train_ppo.py

This script implements the training process for a PPO (Proximal Policy Optimization) model 
to optimize node classification in a graph setting.

Key Features:
- Data loading and preprocessing using `load_data_and_model`.
- Initialization of PPOAgent with an actor-critic structure.
- Training loop that:
  - Processes sequential data using GRU.
  - Selects actions using the policy network.
  - Computes rewards with a custom reward function.
  - Updates the PPO policy using advantage estimation.
- Model validation and checkpointing.
- Final evaluation using accuracy, precision, recall, F1-score, and AUC.

Classes:
- PPOAgent: Implements the PPO algorithm with action selection and policy updates.
- Memory: Stores trajectories for PPO updates.

Functions:
- `custom_reward_function(predicted, label, predicted_distribution=None)`: 
  Defines the reward mechanism for the PPO model.
- `train_ppo_main(config)`: 
  Executes the PPO training loop with hyperparameter tuning.

"""


import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from evaluation.ppo_eval import validate_model, test_model 
from models.wrapper_target import TargetGCN
from models.target_model import GCN
from torch_geometric.datasets import Planetoid
from utils.seed_utils import set_seed
from ppo.memory import Memory, compute_returns_and_advantages
from ppo.ppo_agent import PPOAgent
from ablation.fusion_gru import FusionGRU
from ablation.state_transform_mlp import StateTransformMLP
from embedding.embedding_utils import precompute_all_node_embeddings, k_core_decomposition
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import CitationFull
import networkx as nx
from data_loading.model_loader import load_data_and_model
import numpy as np
import os

def custom_reward_function(predicted, label, predicted_distribution=None):
    reward = 0.0
    if predicted_distribution is not None:
        if predicted_distribution > 0.90:
            reward += -8.0
    if predicted == 1 and label == 0:
        reward += -22.0
    if predicted == 0 and label == 1:
        reward += -18.0
    if predicted == 1 and label == 1:
        reward += 16.0
    if predicted == 0 and label == 0:
        reward += 16.0
    return reward

def train_ppo_main(config):
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_value_list = []

    seed = config.get("seed", 37719)
    K_epochs = config.get("K_epochs", 10)
    batch_size = config.get("batch_size", 16)
    hidden_size = config.get("hidden_size", 196)
    hidden_action_dim = config.get("hidden_action_dim", 16)
    clip_epsilon = config.get("clip_epsilon", 0.30)
    entropy_coef = config.get("entropy_coef", 0.05)
    lr = config.get("lr", 1e-3)
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    num_epochs = config.get("num_epochs", 200)
    save_dir = config.get('save_dir' , None)
    csv_path = config.get("csv_path" , None)
    data_path = config.get("data_path" , "CiteSeer")
    lamb = config.get("lamb" , 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 2


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
        embedding_dim = input_size
        gru = FusionGRU(input_size=input_size, hidden_size=hidden_size).to(device)
        mlp_transform = StateTransformMLP(action_dim, hidden_action_dim, hidden_size).to(device)
        agent = PPOAgent(
            learning_rate=lr,
            batch_size=batch_size,
            K_epochs=K_epochs,
            state_dim=hidden_size,
            action_dim=action_dim,
            gru=gru,
            mlp=mlp_transform,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            device=device
        ).to(device)

        memory = Memory()
        best_val_reward = float('-inf')

        for epoch in range(num_epochs):
            episode_reward = 0.0
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
                hidden_states = gru.process_sequence(all_inputs)
                masked_hidden_states = hidden_states * mask.unsqueeze(-1)
                prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)
                if memory.all_probs:
                    prob_factors[:, :-1] = torch.stack([
                        torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                        for t in range(max_seq_len - 1)
                    ], dim=1).to(device)
                custom_states = (mlp_transform(prob_factors) * masked_hidden_states).detach()
                actions, log_probs, entropies, probs = agent.select_action(
                    custom_states.view(-1, hidden_size)
                )
                actions    = actions.view(len(batch_seqs), max_seq_len)
                log_probs  = log_probs.view(len(batch_seqs), max_seq_len)
                entropies  = entropies.view(len(batch_seqs), max_seq_len)
                probs      = probs.view(len(batch_seqs), max_seq_len, action_dim)
                rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
                dones   = torch.zeros(len(batch_seqs), max_seq_len, device=device)
                batch_predictions = actions.cpu().numpy()
                predicted_distribution = (batch_predictions == 1).mean()
                last_valid_steps = mask.sum(dim=1).long() - 1
                for i in range(len(batch_seqs)):
                    for t in range(last_valid_steps[i] + 1):
                        if mask[i, t] == 1:
                            r = custom_reward_function(
                                actions[i, t].item(),
                                batch_labels[i].item(),
                                predicted_distribution
                            )
                            rewards[i, t] = r
                            episode_reward += r
                    dones[i, last_valid_steps[i]] = 1.0
                memory.store(custom_states, actions, log_probs, rewards, dones,
                            entropy=entropies, masks=mask)
                compute_returns_and_advantages(memory, gamma=gamma, lam=lam)
                agent.update(memory)
                memory.clear()
                
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'gru_state_dict': gru.state_dict(),
            'mlp_transform_state_dict': mlp_transform.state_dict(),
        }, os.path.join(save_dir, f'{seed_now}.pth'))

        agent.eval()
        gru.eval()
        mlp_transform.eval()
        with torch.no_grad():
            accuracy, precision, recall, f1, auc_value = test_model(agent, gru, mlp_transform, test_loader, target_model, data, all_embeddings, hidden_size, device)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            auc_value_list.append(auc_value)
        
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
