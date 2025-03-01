"""
abalation.py

This script implements a training framework using a standard GRU (Gated Recurrent Unit) model
to process sequential data. It includes:
- A StandardGRU model for sequence processing.
- A StateTransformMLP for transforming state representations.
- Functions for precomputing node embeddings.
- A custom reward function for reinforcement learning.
- Training and validation loops for optimizing the PPO (Proximal Policy Optimization) agent.
- Model saving and loading utilities.
- Test procedures including performance evaluation metrics such as accuracy, precision, recall, F1-score, and AUC.

The script supports training variations including:
1. Standard GRU with full model structure.
2. Ablation studies removing MLP transformation.
3. Simple embedding-based training.
"""
import torch
import torch.nn as nn


class StandardGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_state=None):
        output, hidden_n = self.gru(x, hidden_state)
        return output, hidden_n

    def process_sequence(self, inputs, hidden_state=None):
        output, hidden_n = self.forward(inputs, hidden_state)
        return output  

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



def precompute_all_node_embeddings(
    target_model,
    data,
    lamb=1.0
):
  
    all_node_indices = list(range(data.num_nodes))
    all_embeddings = compute_embedding_batch(
        target_model,
        data,
        all_node_indices,
        lamb=lamb
    )
    return all_embeddings

all_embeddings = precompute_all_node_embeddings(
    target_model,
    data,
    lamb=1.0
)

def precompute_simple_embeddings(
    target_model,
    data,
):
  all_node_indices = list(range(data.num_nodes))
  simple_embeddings=simple_embedding_batch(
      target_model,
      data,
      all_node_indices
  )
  return simple_embeddings

simple_embeddings=precompute_simple_embeddings(
    target_model,
    data,
  )

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

all_embeddings = precompute_all_node_embeddings(
    target_model,
    data,
    lamb=2
)



learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.30
K_epochs = 6
entropy_coef = 0.03
hidden_size = 196  
action_dim = 2
num_epochs = 100
w_TP, w_TN, w_FP, w_FN = 2.0, 1.0, 1.0, 2.0
M=10
best_val_reward = float('-inf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = dataset.num_classes
embedding_dim = input_size


gru_s = StandardGRU(input_size, hidden_size)

mlp_transform_s = StateTransformMLP(action_dim, hidden_size)
agent_s = PPOAgent(hidden_size, action_dim, gru_s, mlp_transform_s)
memory = Memory()

gru_s.to(device)
mlp_transform_s.to(device)
agent_s.to(device)

for epoch in range(num_epochs):
    episode_reward = 0.0
    for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
        batch_labels = batch_labels.to(device)  


        batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
        padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)  
        mask = (padded_seqs != 0).float().to(device)  
        max_seq_len = padded_seqs.size(1)
        hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)  
        h_prev = torch.zeros(len(batch_seqs), embedding_dim, device=device)       
        last_valid_steps = mask.sum(dim=1).long() - 1

        all_inputs = []
        
        for t in range(max_seq_len):
            node_indices = padded_seqs[:, t].tolist()
            cur_inputs = all_embeddings[node_indices]
            all_inputs.append(cur_inputs)


        all_inputs = torch.stack(all_inputs, dim=1).to(device)

        
        hidden_states = gru_s.process_sequence(all_inputs)  
        masked_hidden_states = hidden_states * mask.unsqueeze(-1)

        prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)

        if memory.all_probs:  
            prob_factors[:, :-1] = torch.stack([
                torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                for t in range(max_seq_len - 1)
            ], dim=1).to(device)

        custom_states = (mlp_transform_s(prob_factors) * masked_hidden_states).detach()
        actions, log_probs, entropies, probs = agent_s.select_action(custom_states.view(-1, hidden_size))
        actions = actions.view(len(batch_seqs), max_seq_len)
        log_probs = log_probs.view(len(batch_seqs), max_seq_len)
        entropies = entropies.view(len(batch_seqs), max_seq_len)
        probs = probs.view(len(batch_seqs), max_seq_len, action_dim)

        
        rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
        dones = torch.zeros(len(batch_seqs), max_seq_len, device=device)

        batch_predictions = actions.cpu().numpy()
        predicted_distribution = (batch_predictions == 1).mean()  
        for i in range(len(batch_seqs)):
            for t in range(last_valid_steps[i] + 1):
              if mask[i, t] == 1:

                rewards[i, t] = custom_reward_function(actions[i, t].item(), batch_labels[i].item(),predicted_distribution=predicted_distribution)

                episode_reward += rewards[i, t].item()

            dones[i, last_valid_steps[i]] = 1.0  

        
        memory.store(custom_states, actions, log_probs, rewards, dones, entropy=entropies, masks=mask)

        
        compute_returns_and_advantages(memory, gamma=0.99, lam=0.95)

        
        agent_s.update(memory)
        memory.clear()

    print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {episode_reward}")

    
    val_reward, val_accuracy, val_preiciosn, val_recall, val_f1 = validate_model(
        agent_s, gru_s, mlp_transform_s, val_loader, target_model, data
    )
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Reward: {val_reward}, "
          f"Accuracy: {val_accuracy:.2%}, Precision: {val_preiciosn:.2%}, "
          f"Recall: {val_recall:.2%}, F1 Score: {val_f1:.2%}")

    if val_reward > best_val_reward:
        best_val_reward = val_reward
        torch.save({
            'agent_state_dict': agent_s.state_dict(),
            'gru_state_dict': gru_s.state_dict(),
            'mlp_transform_state_dict': mlp_transform_s.state_dict(),
            'epoch': epoch + 1,
            'best_val_reward': best_val_reward
        }, "best_model_sgru.pth")
        print(f"New best model saved with Validation Reward: {val_reward}")



torch.save({
    'agent_state_dict': agent_s.state_dict(),
    'gru_state_dict': gru_s.state_dict(),
    'mlp_transform_state_dict': mlp_transform_s.state_dict(),
    'epoch': num_epochs,
    'final_val_reward': val_reward  
}, "final_model_sgru.pth")
print("Final model saved after training.")


gru_s = StandardGRU(input_size, hidden_size)
mlp_transform_s = StateTransformMLP(action_dim, hidden_size)
agent_s = PPOAgent(hidden_size, action_dim, gru_s, mlp_transform_s)


gru_s.to(device)
mlp_transform_s.to(device)
agent_s.to(device)

import os
def load_final_model(file_path, agent_s, gru_s, mlp_transform_s, device):
    try:
        checkpoint = torch.load(file_path, map_location=device)

        agent_s.load_state_dict(checkpoint['agent_state_dict'])
        gru_s.load_state_dict(checkpoint['gru_state_dict'])
        mlp_transform_s.load_state_dict(checkpoint['mlp_transform_state_dict'])

        epoch = checkpoint.get('epoch', None)
        final_val_reward = checkpoint.get('final_val_reward', None)

        print(f"Model loaded successfully from {file_path}")
        print(f"Loaded Epoch: {epoch}, Final Validation Reward: {final_val_reward}")

        return {
            'epoch': epoch,
            'final_val_reward': final_val_reward
        }
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

final_model_sgru='final_model_sgru.pth'
model_info = load_final_model(final_model_sgru, agent_s, gru_s, mlp_transform_s, device)

if model_info:
    print(f"Model restored from epoch {model_info['epoch']} with final validation reward {model_info['final_val_reward']}")
else:
    print("Failed to load the model.")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def test_model_s(agent, gru, mlp_transform, test_loader, target_model, data):
    agent.eval()  
    gru.eval()
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

                print(f"The actions are {predicted_action}, and the true labels are {true_label}")

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

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    except ValueError:
        print("Unable to compute AUC due to only one class in true labels.")

    print(f"Test Results - Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}, "
          f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")

    return {
        "Total Reward": total_reward,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_value if 'auc_value' in locals() else None
    }

test_model_s(agent_s, gru_s, mlp_transform_s, test_loader, target_model, data)


results_list = []

test_result=test_model_s(agent_s, gru_s, mlp_transform_s, test_loader, target_model, data)
results_list.append(test_result)

def calculate_mean_and_error(metric_values):
    mean = np.mean(metric_values)
    std_error = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))  
    return mean, std_error

print(results_list)

mean_results = {}
for metric in ["Total Reward", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
    metric_values = [r[metric] for r in results_list]
    mean, error = calculate_mean_and_error(metric_values)
    mean_results[metric] = {"Mean": mean, "Error": error}

import json
print("\nTest Results List:")
for i, result in enumerate(results_list):
    print(f"Round {i + 1}: {result}")

print("\nMean Results (± Standard Error):")
for metric, values in mean_results.items():
    print(f"{metric}: {values['Mean']:.4f} ± {values['Error']:.4f}")

with open("train_test_results.json", "w") as f:
    json.dump({"Results List": results_list, "Mean Results": mean_results}, f, indent=4)


def validate_model_no_mlp(agent, gru, mlp_transform, val_loader, target_model, data):
    agent.eval()  
    gru.eval()
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

            custom_states = masked_hidden_states.detach()

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

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_layer = nn.Linear(64, action_dim)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_layer(x)
        state_value = self.value_layer(x)
        return action_logits, state_value


class PPOAgent_no_mlp(nn.Module):
    def __init__(self, state_dim, action_dim, gru):
        super(PPOAgent_no_mlp, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(gru.parameters()),
            lr=learning_rate
        )
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        device = next(self.policy.parameters()).device
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(device)
        else:
            state = torch.tensor(state, dtype=torch.float).to(device)

        with torch.no_grad():
            action_logits, _ = self.policy_old(state)
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs, entropy, probs

    def update(self, memory):
        states = torch.stack(memory.states).view(batch_size, -1, hidden_size).to(device)  
        actions = torch.cat(memory.actions, dim=0)  
        actions = actions.view(batch_size, -1).to(device)  
        log_probs_old = torch.cat(memory.log_probs, dim=0).view(batch_size, -1).to(device)
        returns = memory.returns.view(batch_size,-1).to(device)  
        advantages = memory.advantages.view(batch_size,-1).to(device)  

        for _ in range(K_epochs):
            action_logits, state_values = self.policy(states)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()

            log_probs = log_probs.view_as(advantages)
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values.squeeze(), returns) - \
                   entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []  
        self.returns = []
        self.all_probs = {}  
        self.masks = []  

    def store(self, state, action, log_prob, reward, done, entropy, probs=None, masks=None):
        for i in range(custom_states.size(0)):
            state_seq = custom_states[i]  
            action_seq = actions[i]  
            log_prob_seq = log_probs[i]  
            reward_seq = rewards[i]  
            done_seq = dones[i]  
            mask_seq = masks[i] 

            valid_len = int(mask_seq.sum().item())  

            state_seq = torch.cat([state_seq[:valid_len], torch.zeros(custom_states.size(1) - valid_len, custom_states.size(2), device=state_seq.device)])
            action_seq = torch.cat([action_seq[:valid_len], torch.zeros(actions.size(1) - valid_len, device=action_seq.device)])
            log_prob_seq = torch.cat([log_prob_seq[:valid_len], torch.zeros(log_probs.size(1) - valid_len, device=log_prob_seq.device)])
            reward_seq = torch.cat([reward_seq[:valid_len], torch.zeros(rewards.size(1) - valid_len, device=reward_seq.device)])
            done_seq = torch.cat([done_seq[:valid_len], torch.zeros(dones.size(1) - valid_len, device=done_seq.device)])
            mask_seq = torch.cat([mask_seq[:valid_len], torch.zeros(masks.size(1) - valid_len, device=mask_seq.device)])

            self.states.append(state_seq)
            self.actions.append(action_seq)
            self.log_probs.append(log_prob_seq)
            self.rewards.append(reward_seq)
            self.dones.append(done_seq)
            self.masks.append(mask_seq)

            consistent_shape = all(tensor.shape == self.states[0].shape for tensor in self.states)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []  
        self.returns = []
        self.masks = []  


def compute_returns_and_advantages(memory, gamma=0.99, lam=0.95):
    rewards = torch.stack(memory.rewards, dim=0)  
    dones = torch.stack(memory.dones, dim=0)  
    masks = torch.stack(memory.masks, dim=0)  
    batch_size, max_seq_len = rewards.size()

    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    running_return = torch.zeros(batch_size, device=rewards.device)
    running_advantage = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(max_seq_len)):
        mask_t = masks[:, t]
        reward_t = rewards[:, t]
        done_t = dones[:, t]

        running_return = reward_t + gamma * running_return * (1 - done_t)
        td_error = reward_t + gamma * (returns[:, t + 1] if t + 1 < max_seq_len else 0) * (1 - done_t) - reward_t

        running_return *= mask_t  
        td_error *= mask_t

        returns[:, t] = running_return
        running_advantage = td_error + gamma * lam * running_advantage * (1 - done_t)
        running_advantage *= mask_t
        advantages[:, t] = running_advantage

    memory.returns = returns
    memory.advantages = advantages



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

from torch.nn.utils.rnn import pad_sequence

learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.30
K_epochs = 6
entropy_coef = 0.03
hidden_size = 196  
action_dim = 2
num_epochs = 100
w_TP, w_TN, w_FP, w_FN = 2.0, 1.0, 1.0, 2.0
M=10
best_val_reward = float('-inf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = dataset.num_classes
embedding_dim = input_size

gru_no_mlp = FusionGRU(input_size, hidden_size)


agent_no_mlp = PPOAgent_no_mlp(hidden_size, action_dim, gru_no_mlp)  
memory = Memory()

gru_no_mlp.to(device)
agent_no_mlp.to(device)

for epoch in range(num_epochs):
    episode_reward = 0.0
    for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
        batch_labels = batch_labels.to(device)  

        batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
        padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)
        mask = (padded_seqs != 0).float().to(device)

        max_seq_len = padded_seqs.size(1)
        hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)
        h_prev = torch.zeros(len(batch_seqs), embedding_dim, device=device)
        last_valid_steps = mask.sum(dim=1).long() - 1

        all_inputs = []
        for t in range(max_seq_len):
            node_indices = padded_seqs[:, t].tolist()
            cur_inputs = all_embeddings[node_indices]
            all_inputs.append(cur_inputs)

        all_inputs = torch.stack(all_inputs, dim=1).to(device)

        hidden_states = gru_no_mlp.process_sequence(all_inputs)
        masked_hidden_states = hidden_states * mask.unsqueeze(-1)
        custom_states = masked_hidden_states.detach()

        actions, log_probs, entropies, probs = agent_no_mlp.select_action(
            custom_states.view(-1, hidden_size)
        )
        actions = actions.view(len(batch_seqs), max_seq_len)
        log_probs = log_probs.view(len(batch_seqs), max_seq_len)
        entropies = entropies.view(len(batch_seqs), max_seq_len)
        probs = probs.view(len(batch_seqs), max_seq_len, action_dim)

        rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
        dones = torch.zeros(len(batch_seqs), max_seq_len, device=device)

        batch_predictions = actions.cpu().numpy()
        predicted_distribution = (batch_predictions == 1).mean()  

        for i in range(len(batch_seqs)):
            for t in range(last_valid_steps[i] + 1):
              if mask[i, t] == 1:

                rewards[i, t] = custom_reward_function(actions[i, t].item(), batch_labels[i].item(),predicted_distribution=predicted_distribution)

                episode_reward += rewards[i, t].item()

            dones[i, last_valid_steps[i]] = 1.0  

        memory.store(custom_states, actions, log_probs, rewards, dones, entropy=entropies, masks=mask)
        compute_returns_and_advantages(memory, gamma=0.99, lam=0.95)
        agent_no_mlp.update(memory)
        memory.clear()

    print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {episode_reward}")

    val_reward, val_accuracy, val_precision, val_recall, val_f1 = validate_model_no_mlp(
        agent_no_mlp,
        gru_no_mlp,
        None,  
        val_loader,
        target_model,
        data
    )
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Reward: {val_reward}, "
          f"Accuracy: {val_accuracy:.2%}, Precision: {val_precision:.2%}, "
          f"Recall: {val_recall:.2%}, F1 Score: {val_f1:.2%}")

    if val_reward > best_val_reward:
        best_val_reward = val_reward
        torch.save({
            'agent_state_dict': agent_no_mlp.state_dict(),
            'gru_state_dict': gru_no_mlp.state_dict(),
            'epoch': epoch + 1,
            'best_val_reward': best_val_reward
        }, "best_model_no_mlp.pth")
        print(f"New best model saved (no mlp_transform) with Validation Reward: {val_reward}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def test_model_no_mlp(agent, gru, test_loader, target_model, data):
    agent.eval()  
    gru.eval()

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

            all_inputs = []
            for t in range(max_seq_len):
                node_indices = padded_seqs[:, t].tolist()
                cur_inputs = all_embeddings[node_indices]
                all_inputs.append(cur_inputs)
            all_inputs = torch.stack(all_inputs, dim=1).to(device)

            hidden_states = gru.process_sequence(all_inputs)  
            masked_hidden_states = hidden_states * mask.unsqueeze(-1)

            custom_states = masked_hidden_states.detach()

            actions, probabilities, _, _ = agent.select_action(custom_states.view(-1, hidden_size))
            actions = actions.view(len(batch_seqs), max_seq_len)
            probabilities = probabilities.view(len(batch_seqs), max_seq_len)
          
            for i in range(len(batch_seqs)):
                last_valid_step = (mask[i].sum().long() - 1).item()
                predicted_action = actions[i, last_valid_step].item()
                true_label = batch_labels[i].item()
                predicted_prob = probabilities[i, last_valid_step].item()  

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

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    except ValueError:
        print("Unable to compute AUC due to only one class in true labels.")
        auc_value = None

    print(f"Test Results - Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}, "
          f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")

    return {
        "Total Reward": total_reward,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_value
    }


def calculate_mean_and_error(metric_values):
    mean = np.mean(metric_values)
    std_error = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))  
    return mean, std_error

results_mlp_list = []

test_mlp_result=test_model_no_mlp(agent_no_mlp, gru_no_mlp, test_loader, target_model, data)
results_mlp_list.append(test_mlp_result)
print(results_mlp_list)

mean_results = {}
for metric in ["Total Reward", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
    metric_values = [r[metric] for r in results_mlp_list]
    mean, error = calculate_mean_and_error(metric_values)
    mean_results[metric] = {"Mean": mean, "Error": error}

import json
print("\nTest Results List:")
for i, result in enumerate(results_mlp_list):
    print(f"Round {i + 1}: {result}")

print("\nMean Results (± Standard Error):")
for metric, values in mean_results.items():
    print(f"{metric}: {values['Mean']:.4f} ± {values['Error']:.4f}")

with open("mlp_test_results.json", "w") as f:
    json.dump({"Results List": results_mlp_list, "Mean Results": mean_results}, f, indent=4)


def validate_model_simple(agent, gru, mlp_transform, val_loader, target_model, data):
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
                cur_inputs = simple_embeddings[node_indices]
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
    precision = precision_score(all_true_labels, all_predicted_labels, average='binary')
    recall = recall_score(all_true_labels, all_predicted_labels, average='binary')
    f1 = f1_score(all_true_labels, all_predicted_labels, average='binary')

    return total_reward, accuracy, precision, recall, f1

from torch.nn.utils.rnn import pad_sequence
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.30
K_epochs = 6
entropy_coef = 0.03
hidden_size = 196  
action_dim = 2
num_epochs = 100
w_TP, w_TN, w_FP, w_FN = 2.0, 1.0, 1.0, 2.0
best_val_reward = float('-inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size=dataset.num_classes
embedding_dim=input_size
gru_simple = FusionGRU(input_size, hidden_size)
mlp_transform_simple = StateTransformMLP(action_dim, hidden_size)
agent_simple = PPOAgent(hidden_size, action_dim, gru_simple, mlp_transform_simple)
memory = Memory()

gru_simple.to(device)
mlp_transform_simple.to(device)
agent_simple.to(device)

for epoch in range(num_epochs):
    episode_reward = 0.0
    for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
        batch_labels = batch_labels.to(device) 

        batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
        padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)  
        mask = (padded_seqs != 0).float().to(device)  

        max_seq_len = padded_seqs.size(1)
        hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)
        h_prev = torch.zeros(len(batch_seqs), embedding_dim, device=device)
        last_valid_steps = mask.sum(dim=1).long() - 1

        all_inputs = []

        for t in range(max_seq_len):
            node_indices = padded_seqs[:, t].tolist()
            cur_inputs = simple_embeddings[node_indices]
            all_inputs.append(cur_inputs)

        all_inputs = torch.stack(all_inputs, dim=1).to(device)  

        hidden_states = gru_simple.process_sequence(all_inputs)  
        masked_hidden_states = hidden_states * mask.unsqueeze(-1)  

        prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)  
        if memory.all_probs:  
            prob_factors[:, :-1] = torch.stack([
                torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                for t in range(max_seq_len - 1)
            ], dim=1).to(device)
        custom_states = (mlp_transform_simple(prob_factors) * masked_hidden_states).detach()
        actions, log_probs, entropies, probs = agent_simple.select_action(custom_states.view(-1, hidden_size))
        actions = actions.view(len(batch_seqs), max_seq_len)
        log_probs = log_probs.view(len(batch_seqs), max_seq_len)
        entropies = entropies.view(len(batch_seqs), max_seq_len)
        probs = probs.view(len(batch_seqs), max_seq_len, action_dim)

       
        rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
        dones = torch.zeros(len(batch_seqs), max_seq_len, device=device)

        batch_predictions = actions.cpu().numpy()
        predicted_distribution = (batch_predictions == 1).mean() 

        for i in range(len(batch_seqs)):
            for t in range(last_valid_steps[i] + 1):
              if mask[i, t] == 1:

                rewards[i, t] = custom_reward_function(actions[i, t].item(), batch_labels[i].item(),predicted_distribution=predicted_distribution)

                episode_reward += rewards[i, t].item()

            dones[i, last_valid_steps[i]] = 1.0 


        memory.store(custom_states, actions, log_probs, rewards, dones, entropy=entropies, masks=mask)
        compute_returns_and_advantages(memory, gamma=0.99, lam=0.95)
        agent_simple.update(memory)
        memory.clear()
    print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {episode_reward}")

    val_reward, val_accuracy, val_preiciosn, val_recall, val_f1 = validate_model_simple(agent_simple, gru_simple, mlp_transform_simple, val_loader, target_model, data)

    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Reward: {val_reward}, Accuracy: {val_accuracy:.2%},"f"Precision: {val_preiciosn:.2%}, Recall: {val_recall:.2%}, F1 Score: {val_f1:.2%}")
    if val_reward > best_val_reward:
        best_val_reward = val_reward
        torch.save({
            'agent_state_dict': agent_simple.state_dict(),
            'gru_state_dict': gru_simple.state_dict(),
            'mlp_transform_state_dict': mlp_transform_simple.state_dict(),
            'epoch': epoch + 1,
            'best_val_reward': best_val_reward
        }, "best_model_simple.pth")
        print(f"New best model saved with Validation Reward: {val_reward}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def test_model_simple(agent, gru, mlp_transform, test_loader, target_model, data):
    agent.eval()  
    gru.eval()
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
                cur_inputs = simple_embeddings[node_indices]
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

                print(f"The actions are {predicted_action}, and the true labels are {true_label}")

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

    
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    except ValueError:
        print("Unable to compute AUC due to only one class in true labels.")

    print(f"Test Results - Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}, "
          f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")

    return {
        "Total Reward": total_reward,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_value if 'auc_value' in locals() else None
    }


checkpoint = torch.load("best_model_simple.pth", map_location=device)
agent_simple.load_state_dict(checkpoint['agent_state_dict'])
gru_simple.load_state_dict(checkpoint['gru_state_dict'])
mlp_transform_simple.load_state_dict(checkpoint['mlp_transform_state_dict'])
agent_simple.eval()
gru_simple.eval()
mlp_transform_simple.eval()


test_model_simple(agent_simple, gru_simple, mlp_transform_simple, test_loader, target_model, data)
test_model_simple(agent_simple, gru_simple, mlp_transform_simple, new_test_loader, target_model, data)



def calculate_mean_and_error(metric_values):
    mean = np.mean(metric_values)
    std_error = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))  
    return mean, std_error

results_simple_list = []

for _ in range(10):
  learning_rate = 3e-4
  gamma = 0.99
  clip_epsilon = 0.30
  K_epochs = 6
  entropy_coef = 0.03
  hidden_size = 196  
  action_dim = 2
  num_epochs = 100
  w_TP, w_TN, w_FP, w_FN = 2.0, 1.0, 1.0, 2.0
  best_val_reward = float('-inf')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_size=dataset.num_classes
  embedding_dim=input_size
  gru_simple = FusionGRU(input_size, hidden_size)
  mlp_transform_simple = StateTransformMLP(action_dim, hidden_size)
  agent_simple = PPOAgent(hidden_size, action_dim, gru_simple, mlp_transform_simple)
  memory = Memory()

  gru_simple.to(device)
  mlp_transform_simple.to(device)
  agent_simple.to(device)

  for epoch in range(num_epochs):
      episode_reward = 0.0
      for batch_idx, (batch_seqs, batch_labels) in enumerate(train_loader):
          batch_labels = batch_labels.to(device)  
         
          batch_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_seqs]
          padded_seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0)  
          mask = (padded_seqs != 0).float().to(device) 

          max_seq_len = padded_seqs.size(1)
          hidden_states = torch.zeros(len(batch_seqs), hidden_size, device=device)
          h_prev = torch.zeros(len(batch_seqs), embedding_dim, device=device)
          last_valid_steps = mask.sum(dim=1).long() - 1

          all_inputs = []

          for t in range(max_seq_len):
              node_indices = padded_seqs[:, t].tolist()
              cur_inputs = simple_embeddings[node_indices]
              all_inputs.append(cur_inputs)

          all_inputs = torch.stack(all_inputs, dim=1).to(device)  

          hidden_states = gru_simple.process_sequence(all_inputs)  
          masked_hidden_states = hidden_states * mask.unsqueeze(-1)  

          prob_factors = torch.ones(len(batch_seqs), max_seq_len, action_dim, device=device)  
          if memory.all_probs: 
              prob_factors[:, :-1] = torch.stack([
                  torch.tensor(memory.all_probs.get(t, [1.0] * action_dim))
                  for t in range(max_seq_len - 1)
              ], dim=1).to(device)
          custom_states = (mlp_transform_simple(prob_factors) * masked_hidden_states).detach()
          actions, log_probs, entropies, probs = agent_simple.select_action(custom_states.view(-1, hidden_size))
          actions = actions.view(len(batch_seqs), max_seq_len)
          log_probs = log_probs.view(len(batch_seqs), max_seq_len)
          entropies = entropies.view(len(batch_seqs), max_seq_len)
          probs = probs.view(len(batch_seqs), max_seq_len, action_dim)

          
          rewards = torch.zeros(len(batch_seqs), max_seq_len, device=device)
          dones = torch.zeros(len(batch_seqs), max_seq_len, device=device)

          batch_predictions = actions.cpu().numpy()
          predicted_distribution = (batch_predictions == 1).mean()  

          for i in range(len(batch_seqs)):
              for t in range(last_valid_steps[i] + 1):
                if mask[i, t] == 1:

                  rewards[i, t] = custom_reward_function(actions[i, t].item(), batch_labels[i].item(),predicted_distribution=predicted_distribution)

                  episode_reward += rewards[i, t].item()

              dones[i, last_valid_steps[i]] = 1.0  


          memory.store(custom_states, actions, log_probs, rewards, dones, entropy=entropies, masks=mask)
          compute_returns_and_advantages(memory, gamma=0.99, lam=0.95)
          agent_simple.update(memory)
          memory.clear()
      print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {episode_reward}")

      val_reward, val_accuracy, val_preiciosn, val_recall, val_f1 = validate_model_simple(agent_simple, gru_simple, mlp_transform_simple, val_loader, target_model, data)

      print(f"Epoch {epoch + 1}/{num_epochs} - Validation Reward: {val_reward}, Accuracy: {val_accuracy:.2%},"f"Precision: {val_preiciosn:.2%}, Recall: {val_recall:.2%}, F1 Score: {val_f1:.2%}")
      if val_reward > best_val_reward:
          best_val_reward = val_reward
          torch.save({
              'agent_state_dict': agent_simple.state_dict(),
              'gru_state_dict': gru_simple.state_dict(),
              'mlp_transform_state_dict': mlp_transform_simple.state_dict(),
              'epoch': epoch + 1,
              'best_val_reward': best_val_reward
          }, "best_model_simple.pth")
          print(f"New best model saved with Validation Reward: {val_reward}")
  test_simple_result=test_model_simple(agent_simple, gru_simple, mlp_transform_simple, test_loader, target_model, data)
  results_simple_list.append(test_simple_result)
  print(results_simple_list)

test_simple_result=test_model_simple(agent_simple, gru_simple, mlp_transform_simple, test_loader, target_model, data)
results_simple_list.append(test_simple_result)
print(results_simple_list)

mean_results = {}
for metric in ["Total Reward", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
    metric_values = [r[metric] for r in results_simple_list]
    mean, error = calculate_mean_and_error(metric_values)
    mean_results[metric] = {"Mean": mean, "Error": error}

import json
print("\nTest Results List:")
for i, result in enumerate(results_simple_list):
    print(f"Round {i + 1}: {result}")

print("\nMean Results (± Standard Error):")
for metric, values in mean_results.items():
    print(f"{metric}: {values['Mean']:.4f} ± {values['Error']:.4f}")


with open("simple_test_results.json", "w") as f:
    json.dump({"Results List": results_simple_list, "Mean Results": mean_results}, f, indent=4)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def test_model(agent, gru, mlp_transform, test_loader, target_model, data):
    agent.eval()  
    gru.eval()
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

                print(f"The actions are {predicted_action}, and the true labels are {true_label}")

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

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    except ValueError:
        print("Unable to compute AUC due to only one class in true labels.")

    print(f"Test Results - Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}, "
          f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")

    return {
        "Total Reward": total_reward,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_value if 'auc_value' in locals() else None
    }
test_model(agent, gru, mlp_transform, test_loader, target_model, data)


test_model(agent, gru, mlp_transform, test_loader, target_model, data)
test_model(agent, gru, mlp_transform, new_test_loader, target_model, data)


final_model_path = "final_model.pth"

torch.save({
    'agent_state_dict': agent.state_dict(),
    'gru_state_dict': gru.state_dict(),
    'mlp_transform_state_dict': mlp_transform.state_dict(),
    'hyperparameters': {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'clip_epsilon': clip_epsilon,
        'K_epochs': K_epochs,
        'entropy_coef': entropy_coef,
        'hidden_size': hidden_size,
        'action_dim': action_dim,
        'num_epochs': num_epochs,
    }
}, final_model_path)

print(f"Final model saved at {final_model_path}")


import os
def load_model(agent, gru, mlp_transform, file_path, device):
    checkpoint = torch.load(file_path, map_location=device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    gru.load_state_dict(checkpoint['gru_state_dict'])
    mlp_transform.load_state_dict(checkpoint['mlp_transform_state_dict'])
    hyperparameters = checkpoint.get('hyperparameters', {})
    print(f"Final model loaded from {file_path}")
    return hyperparameters


final_model_hM="CiteSeer_final_model.pth"


loaded_hyperparameters = load_model(agent, gru, mlp_transform, final_model_hM, device)

test_model(agent, gru, mlp_transform, test_loader, target_model, data)

test_model(agent, gru, mlp_transform, new_test_loader, target_model, data)

checkpoint = torch.load("best_model.pth", map_location=device)
agent.load_state_dict(checkpoint['agent_state_dict'])
gru.load_state_dict(checkpoint['gru_state_dict'])
mlp_transform.load_state_dict(checkpoint['mlp_transform_state_dict'])
agent.eval()
gru.eval()
mlp_transform.eval()

print(f"Model loaded from epoch {checkpoint.get('epoch', 0)} with best validation reward: {checkpoint.get('best_val_reward', float('-inf'))}")


test_model(agent, gru, mlp_transform, test_loader, target_model, data)