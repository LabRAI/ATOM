"""
ppo_agent.py

This script implements the PPO (Proximal Policy Optimization) agent for reinforcement learning.
It includes:
- A `PolicyNetwork` class to define the actor-critic architecture.
- A `PPOAgent` class that manages action selection, training updates, and optimization.
- The `compute_returns_and_advantages` function using Generalized Advantage Estimation (GAE).

Classes:
- PolicyNetwork:
  - Defines the policy and value function for PPO.
  - Uses a two-layer fully connected architecture.

- PPOAgent:
  - Manages training using PPO loss functions.
  - Handles action selection, advantage computation, and gradient updates.

Functions:
- compute_returns_and_advantages(memory, gamma=0.99, lam=0.95):
  Computes the returns and advantages for PPO training using GAE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical



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



class PPOAgent(nn.Module):
    def __init__(self, learning_rate, batch_size, K_epochs, state_dim, action_dim, gru, mlp, clip_epsilon, entropy_coef, device):
        super(PPOAgent, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(gru.parameters()) + list(mlp.parameters()),
            lr=learning_rate
        )
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.device = device
        self.hidden_size = state_dim
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

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
        states = torch.stack(memory.states).view(self.batch_size, -1, self.hidden_size).to(self.device)  
        actions = torch.cat(memory.actions, dim=0)  
        actions = actions.view(self.batch_size, -1).to(self.device)  
        log_probs_old = torch.cat(memory.log_probs, dim=0).view(self.batch_size, -1).to(self.device)
        returns = memory.returns.view(self.batch_size,-1).to(self.device) 
        advantages = memory.advantages.view(self.batch_size,-1).to(self.device)  

        for _ in range(self.K_epochs):
            action_logits, state_values = self.policy(states)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()

            log_probs = log_probs.view_as(advantages)
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values.squeeze(), returns) - \
                   self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

def compute_returns_and_advantages(memory, gamma=0.99, lam=0.95):
    rewards = torch.stack(memory.rewards, dim=0).squeeze(-1)
    dones = torch.stack(memory.dones, dim=0).squeeze(-1)

    batch_size = rewards.size(0)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    running_return = 0.0
    running_adv = 0.0

    for t in reversed(range(batch_size)):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return
        advantages[t] = returns[t] - 0  
    memory.returns = returns
    memory.advantages = advantages
