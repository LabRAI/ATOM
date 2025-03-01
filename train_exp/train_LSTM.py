"""
train_lstm.py

This script implements the training process for the StandardLSTM model using PPO (Proximal Policy Optimization).
It includes:
- Data loading and preprocessing.
- Model initialization for StandardLSTM, FusionGRU, and associated components.
- Training loop with sequence processing, action selection, and reward computation.
- Hyperparameter configurations such as batch size, learning rate, and entropy coefficient.
- Model validation and saving the best model based on validation reward.
- Evaluation of model performance using metrics like accuracy, precision, recall, F1-score, and AUC.

"""

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.LSTM import StandardLSTM,custom_reward_function,test_model_lstm,lstm_validate_model
from ppo.ppo_agent import PPOAgent
from models.MLP import StateTransformMLP
from utils.seed_utils import set_seed
from data_loading.model_loader import load_data_and_model
from embedding.embedding_utils import precompute_all_node_embeddings
from ppo.memory import Memory, compute_returns_and_advantages
import numpy as np
import os
from ablation.fusion_gru import FusionGRU

def train_lstm(config):
    seed = config.get("seed", 37719)
    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 15)
    learning_rate = config.get("learning_rate", 3e-2)
    lamb = config.get("lamb", 1)
    K_epochs = config.get("K_epochs", 10)
    gamma = config.get("gamma", 0.99)
    csv_path = config.get("csv_path" , None)
    clip_epsilon = config.get("clip_epsilon", 0.30)
    entropy_coef = config.get("entropy_coef", 0.05)
    data_path = config.get("data_path" , "CiteSeer")
    save_dir = config.get('save_dir' , None)
    hidden_size = config.get("hidden_size", 336)
    action_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rate = config.get("rate" , 0.25)

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
        lstm_s = StandardLSTM(input_size, hidden_size).to(device)
        mlp_transform_s = StateTransformMLP(action_dim, hidden_size).to(device)
        gru = FusionGRU(input_size=input_size, hidden_size=hidden_size).to(device)
        agent_s = PPOAgent(
            learning_rate=learning_rate,
            batch_size=batch_size,
            K_epochs=K_epochs,
            state_dim=hidden_size,
            action_dim=action_dim,
            gru=gru,
            mlp=mlp_transform_s,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            device=device
        ).to(device)
        memory = Memory()

        optimizer = optim.Adam(agent_s.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            best_val_reward = float('-inf')
            episode_reward = 0.0
            for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
                batch_labels = batch_labels.to(device)

                batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
                padded_seq = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
                max_seq_len = padded_seq.size(1)
                num_steps = int(max_seq_len * rate)
        
                truncated_seq = padded_seq[:, :num_steps]
                mask = (truncated_seq != 0).float().to(device)


                last_valid_steps = mask.sum(dim=1).long() - 1

                all_inputs = []
                for t in range(num_steps):
                    node_indices = truncated_seq[:, t].tolist()
                    cur_inputs = all_embeddings[node_indices]
                    all_inputs.append(cur_inputs)

                all_inputs = torch.stack(all_inputs, dim=1).to(device)

                hidden_states = lstm_s.process_sequence(all_inputs)
                masked_hidden_states = hidden_states * mask.unsqueeze(-1)

                prob_factors = torch.ones(len(batch_seqs), num_steps, action_dim, device=device)

                if memory.all_probs:
                    prob_factors[:, :-1] = torch.stack([
                        torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                        for t in range(num_steps - 1)
                    ], dim=1).to(device)

                custom_states = (mlp_transform_s(prob_factors) * masked_hidden_states).detach()
                actions, log_probs, entropies, probs = agent_s.select_action(custom_states.view(-1, hidden_size))

                actions = actions.view(len(batch_seqs), num_steps)
                log_probs = log_probs.view(len(batch_seqs), num_steps)
                entropies = entropies.view(len(batch_seqs), num_steps)
                probs = probs.view(len(batch_seqs), num_steps, action_dim)

                rewards = torch.zeros(len(batch_seqs), num_steps, device=device)
                dones = torch.zeros(len(batch_seqs), num_steps, device=device)

                batch_predictions = actions.cpu().numpy()
                predicted_distribution = (batch_predictions == 1).mean()

                for i in range(len(batch_seqs)):
                    for t in range(last_valid_steps[i] + 1):
                        if mask[i, t] == 1:
                            rewards[i, t] = custom_reward_function(actions[i, t].item(), batch_labels[i].item(), predicted_distribution)
                            episode_reward += rewards[i, t].item()

                    dones[i, last_valid_steps[i]] = 1.0

                memory.store(custom_states, actions, log_probs, rewards, dones, entropy=entropies, masks=mask)

                compute_returns_and_advantages(memory, gamma=gamma, lam=0.95)

                agent_s.update(memory)
                memory.clear()
            
            # verfication
            # val_reward, val_accuracy, val_precision, val_recall, val_f1 = lstm_validate_model(
            #     agent_s, lstm_s, mlp_transform_s, val_loader, target_model, data
            # )

            # if val_reward > best_val_reward:
            #     best_val_reward = val_reward
            #     torch.save({
            #         'agent_state_dict': agent_s.state_dict(),
            #         'lstm_state_dict': lstm_s.state_dict(),
            #         'mlp_transform_state_dict': mlp_transform_s.state_dict(),
            #         'epoch': epoch + 1,
            #         'best_val_reward': best_val_reward
            #     }, f"best_lstm_model_{lamb}.pth")

        with torch.no_grad():
            accuracy, precision, recall, f1, auc_value = test_model_lstm(
                agent_s, lstm_s, mlp_transform_s, test_loader, target_model, data, hidden_size, all_embeddings, action_dim, device
            )
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            auc_value_list.append(auc_value)
        torch.save({
            'lstm_s': lstm_s.state_dict(),
            'mlp_transform_s': mlp_transform_s.state_dict(),
            'agent_s': agent_s.state_dict(),
        }, os.path.join(save_dir, f'{seed_now}.pth'))

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_value_list.append(auc_value)

    def print_stats(name, values):
        print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

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