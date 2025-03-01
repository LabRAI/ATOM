"""
ppo_config.py

This script defines the hyperparameter search space for training a PPO-based model on various graph datasets
using Ray Tune. Each dataset has its own configuration, specifying:
- Learning rates, batch sizes, and number of training epochs.
- PPO-specific parameters such as hidden layer sizes, action dimensions, gamma, lambda, and entropy coefficient.
- Paths to CSV files containing processed attack data for each dataset.
- Seed values for reproducibility.

Datasets included:
- CiteSeer, Cora, PubMed, Cora_ML, Cornell, Wisconsin

Ray Tune will sample different combinations of these hyperparameters to optimize PPO model performance.
"""

from ray import tune

ppo_config = {
        "CiteSeer":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 1,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/attack_CiteSeer_new.csv",
        "saved_name": "CiteSeer",
        "data_path": "CiteSeer",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },
    "Cora":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 100,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/attack_Cora.csv",
        "saved_name": "Cora",
        "data_path": "Cora",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },
    "PubMed":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 100,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/attack_PubMed.csv",
        "saved_name": "PubMed",
        "data_path": "PubMed",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },
    "Cora_ML":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 100,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/new_attack_Cora_ML.csv",
        "saved_name": "Cora_ML",
        "data_path": "Cora_ML",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },
    "Cornell":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 100,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/new_attack_Corenell.csv",
        "saved_name": "Cornell",
        "data_path": "Cornell",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },
    "Wisconsin":{
        'tp': 'ppo',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,  
        "hidden_size": tune.choice([64, 128, 196, 256, 512]),
        "hidden_action_dim": tune.choice([8, 16, 32, 24, 64, 48]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "num_epochs": 100,  
        "K_epochs": 6,      
        "gamma": 0.99,       
        "csv_path": "./csv_data/new_attack_Wisconsin.csv",
        "saved_name": "Wisconsin",
        "data_path": "Wisconsin",
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]),
    },

}