"""
lstm_config.py

This script defines the hyperparameter search space for training an LSTM-based PPO model on various graph datasets
using Ray Tune. Each dataset has its own configuration, specifying:
- Learning rates, batch sizes, and number of training epochs.
- Hidden size variations for the LSTM network.
- PPO-specific parameters such as clipping epsilon, entropy coefficient, and lambda.
- Paths to CSV files containing processed attack data for each dataset.
- Seed values for reproducibility.

Datasets included:
- CiteSeer, Cora, PubMed, Cora_ML, Cornell, Wisconsin

Ray Tune will sample different combinations of these hyperparameters to optimize training performance.
"""

from ray import tune

lstm_config = {
        "CiteSeer":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_CiteSeer_new.csv",
        "saved_name": "CiteSeer",
        "data_path": "CiteSeer",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "Cora":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_Cora.csv",
        "saved_name": "Cora",
        "data_path": "Cora",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "PubMed":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/attack_PubMed.csv",
        "saved_name": "PubMed",
        "data_path": "PubMed",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },
    "Cora_ML":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Cora_ML.csv",
        "saved_name": "Cora_ML",
        "data_path": "Cora_ML",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,

    },
    "Cornell":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Corenell.csv",
        "saved_name": "Cornell",
        "data_path": "Cornell",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,

    },
    "Wisconsin":{
        'tp': 'lstm',
        "seed": [40, 43, 37719, 1005, 2005, 913],
        "lr": tune.choice([5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]),
        "batch_size": 16,
        "num_epochs": tune.choice([2, 3, 4, 5, 8, 10, 15, 20]),
        "K_epochs": 6,
        "gamma": 0.99,
        "csv_path": "./csv_data/new_attack_Wisconsin.csv",
        "saved_name": "Wisconsin",
        "data_path": "Wisconsin",
        "hidden_size": tune.choice([16, 24, 32, 64, 96, 192, 336]),
        "clip_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "entropy_coef": tune.choice([0.01, 0.02, 0.03, 0.1]),
        "mlp_output_size": 2,
        "lam": tune.choice([0.95, 0.85]),
        "lamb": tune.choice([1]),
        "n_runs": 10,
    },

}