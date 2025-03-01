"""
memory.py

This script implements the memory buffer used for PPO (Proximal Policy Optimization) reinforcement learning.
It includes:
- The `Memory` class to store episode trajectories, including states, actions, log probabilities, rewards, entropy, and masks.
- Functions to compute returns and advantages using Generalized Advantage Estimation (GAE).
- A custom reward function for reinforcement learning with modified penalties for false positives and false negatives.

Classes:
- Memory:
  - Stores and manages trajectory data for PPO training.
  - Handles padded sequences and masks for variable-length episodes.

Functions:
- compute_returns_and_advantages(memory, gamma=0.99, lam=0.95):
  Computes the returns and advantages for PPO training using GAE.

- custom_reward_function(predicted, label):
  Defines a custom reward scheme with penalties for false positives and false negatives.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

    def store(self, custom_states, action, log_prob, reward, done, entropy, probs=None, masks=None):
        for i in range(custom_states.size(0)):
            state_seq = custom_states[i]  
            action_seq = action[i]  
            log_prob_seq = log_prob[i]  
            reward_seq = reward[i]  
            done_seq = done[i]  
            mask_seq = masks[i]  

            valid_len = int(mask_seq.sum().item())  

            state_seq = torch.cat([state_seq[:valid_len], torch.zeros(custom_states.size(1) - valid_len, custom_states.size(2), device=state_seq.device)])
            action_seq = torch.cat([action_seq[:valid_len], torch.zeros(action.size(1) - valid_len, device=action_seq.device)])
            log_prob_seq = torch.cat([log_prob_seq[:valid_len], torch.zeros(log_prob.size(1) - valid_len, device=log_prob_seq.device)])
            reward_seq = torch.cat([reward_seq[:valid_len], torch.zeros(reward.size(1) - valid_len, device=reward_seq.device)])
            done_seq = torch.cat([done_seq[:valid_len], torch.zeros(done.size(1) - valid_len, device=done_seq.device)])
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

def custom_reward_function(predicted, label):
    if predicted == 1 and label == 0: 
        return -22.0  
    if predicted == 0 and label == 1:  
        return -18.0 
    if predicted == 1 and label == 1:  
        return 16.0  
    if predicted == 0 and label == 0: 
        return 16.0  

