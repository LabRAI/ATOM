"""
mlp_config.py

This script defines the hyperparameter search space for training an MLP-based model on various graph datasets
using Ray Tune. Each dataset has its own configuration, specifying:
- Learning rates, batch sizes, and number of training epochs.
- MLP-specific parameters such as the number of hidden layers and their sizes.
- PPO-specific parameters such as gamma, lambda, and entropy coefficient.
- Paths to CSV files containing processed attack data for each dataset.
- Seed values for reproducibility.

Datasets included:
- CiteSeer, Cora, PubMed, Cora_ML, Cornell, Wisconsin

Ray Tune will sample different combinations of these hyperparameters to optimize training performance.
"""

from ray import tune


mlp_config = {
        "CiteSeer":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_CiteSeer_new.csv",
        "saved_name": "CiteSeer",
        "data_path": "CiteSeer",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "Cora":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_Cora.csv",
        "saved_name": "Cora",
        "data_path": "Cora",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "PubMed":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_PubMed.csv",
        "saved_name": "PubMed",
        "data_path": "PubMed",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "Cora_ML":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Cora_ML.csv",
        "saved_name": "Cora_ML",
        "data_path": "Cora_ML",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,

    },
    "Cornell":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Corenell.csv",
        "saved_name": "Cornell",
        "data_path": "Cornell",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,

    },
    "Wisconsin":{
        'tp': 'mlp',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Wisconsin.csv",
        "saved_name": "Wisconsin",
        "data_path": "Wisconsin",
        "mlp_hidden_size1": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_hidden_size2": tune.choice([8, 12, 16, 24, 32, 64]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },

}