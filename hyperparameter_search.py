"""
hyperparameter_search_ray_tune.py

This script performs hyperparameter tuning using Ray Tune for different model architectures including PPO, MLP, RNN, LSTM, and Transformer.
It includes:
- Configuration loading for different models.
- Training function execution based on the selected model type.
- Logging of test statistics including AUC, accuracy, precision, recall, and F1-score.
- Automatic model selection and metric-based hyperparameter optimization using ASHAScheduler.
- Saving results for future analysis.
"""

import os
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from train_exp.train_ppo import train_ppo_main
from train_exp.train_mlp import train_mlp
from train_exp.train_RNN import train_rnn
from train_exp.train_LSTM import train_lstm
from train_exp.train_transformer import train_transformer
from hyperparameter_config.ppo_config import ppo_config
from hyperparameter_config.mlp_config import mlp_config
from hyperparameter_config.rnn_config import rnn_config
from hyperparameter_config.lstm_config import lstm_config
from hyperparameter_config.transformer_config import transformer_config
import argparse
import uuid  
import json
from ray.train import report
from pathlib import Path



def train_ppo_tune(config, checkpoint_dir=None):
    try:
        script_dir = Path(__file__).resolve()
        parent_dir = script_dir.parent
    except NameError:
        parent_dir = Path.cwd().parent
    os.chdir(parent_dir)

    unique_id = str(uuid.uuid4())
    tp = config.get('tp', 'ppo')
    save_dir = os.path.join(f"./Data_logs/{tp}/{config['saved_name']}", unique_id)
    print(save_dir)
    config["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    if tp == 'mlp':
        test_stats = train_mlp(config)
    elif tp == 'rnn':
        test_stats = train_rnn(config)
    elif tp == 'lstm':
        test_stats = train_lstm(config)
    elif tp == 'transformer':
        test_stats = train_transformer(config)
    elif tp == 'ppo':
        test_stats = train_ppo_main(config)

    report_metrics = {
        "test_AUC": test_stats.get("auc", 0),
        "test_accuracy": test_stats.get("accuracy", 0),
        "test_precision": test_stats.get("precision", 0),
        "test_recall": test_stats.get("recall", 0),
        "test_f1": test_stats.get("f1_score", 0),
    }

    log_data = {
        "trial_id": unique_id,
        "config": config,
        "test_stats": test_stats
    }
    rate = config.get("rate" , 0.25)
    
    with open(f"./Data_logs_{rate}/{config['saved_name']}_ray_tune_search_results.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

    report(report_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training program")
    parser.add_argument(
        '--tp',
        type=str,
        default='transformer',
        help="'transformer', 'lstm', 'mlp' 等"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CiteSeer',
        help="'CiteSeer'等"
    )
    parser.add_argument(
        '--rate',
        type=float,
        default='1',
        help=""
    )
    args = parser.parse_args()
    tp = args.tp
    rate = args.rate

    dataset = args.dataset

    config_zz = {
        'ppo': ppo_config,
        'mlp': mlp_config,
        'rnn': rnn_config,
        'lstm': lstm_config,
        'transformer': transformer_config,
    }
    os.makedirs(f"Data_logs_{rate}/{tp}", exist_ok=True)
    config = config_zz[tp][dataset]
    config['tp'] = tp
    config['rate'] = rate

    scheduler = ASHAScheduler(
        metric="test_AUC",
        mode="max",
        max_t=200,
        grace_period=10,
        reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=["test_AUC", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    )
    analysis = tune.run(
        train_ppo_tune,
        resources_per_trial={"cpu": 2, "gpu": 0.1}, 
        config=config,
        num_samples=20,  
        scheduler=scheduler,
        progress_reporter=reporter,
        name=f"{tp}_hyperparameter_search",
        max_concurrent_trials=1
    )

    best_trial = analysis.get_best_trial("test_AUC", "max", "last")
    tune.shutdown()
