# ATOM: A Framework of Detecting Query-Based Model Extraction Attacks for Graph Neural Networks

## Environment

The experiments are carried out on a server equipped with an AMD EPYC 7443 24-Core Processor and four 4090 graphics cards.

Library details can be found in requirements.txt.

CUDA Version: 12.4

## Config

The search space for hyperparameters is located under the ./hyperparameter_config directory.
```bash
./mlp_config.py
./rnn_config.py
./lstm_config.py
./transformer_config.py
./ppo_config.py ## (Ours)
```

## Usage

### For testing the CiteSeer dataset
```bash
python hyperparameter_search.py --tp mlp --dataset CiteSeer
python hyperparameter_search.py --tp rnn --dataset CiteSeer
python hyperparameter_search.py --tp lstm --dataset CiteSeer
python hyperparameter_search.py --tp transformer --dataset CiteSeer
python hyperparameter_search.py --tp ppo --dataset CiteSeer ## (Ours)
```

### For testing the Cora dataset
```bash
python hyperparameter_search.py --tp mlp --dataset Cora
python hyperparameter_search.py --tp rnn --dataset Cora
python hyperparameter_search.py --tp lstm --dataset Cora
python hyperparameter_search.py --tp transformer --dataset Cora
python hyperparameter_search.py --tp ppo --dataset Cora ## (Ours)
```

### For testing the PubMed dataset
```bash
python hyperparameter_search.py --tp mlp --dataset PubMed
python hyperparameter_search.py --tp rnn --dataset PubMed
python hyperparameter_search.py --tp lstm --dataset PubMed
python hyperparameter_search.py --tp transformer --dataset PubMed
python hyperparameter_search.py --tp ppo --dataset PubMed ## (Ours)
```

### For testing the Cornell dataset

```bash
python hyperparameter_search.py --tp mlp --dataset Cornell
python hyperparameter_search.py --tp rnn --dataset Cornell
python hyperparameter_search.py --tp lstm --dataset Cornell
python hyperparameter_search.py --tp transformer --dataset Cornell
python hyperparameter_search.py --tp ppo --dataset Cornell ## (Ours)
```

### For testing the Wisconsin dataset
```bash
python hyperparameter_search.py --tp mlp --dataset Wisconsin
python hyperparameter_search.py --tp rnn --dataset Wisconsin
python hyperparameter_search.py --tp lstm --dataset Wisconsin
python hyperparameter_search.py --tp transformer --dataset Wisconsin
python hyperparameter_search.py --tp ppo --dataset Wisconsin ## (Ours)
```

## Logs
After running the corresponding test commands mentioned above, a Data_logs folder will be generated in the directory. Under the ./Data_logs directory, files ending with .log contain the output of the corresponding test results. The --tp *model (the model specified by the user) and ./Data_logs/*model correspond to the saved model parameters of the respective results, which facilitates debugging and testing the model weights with better performance.

The final ./Data_logs should be as follows:
```bash
./Data_logs/
CiteSeer_ray_tune_search_results.log
Cora_ray_tune_search_results.log
PubMed_ray_tune_search_results.log
Cornell_ray_tune_search_results.log
Wisconsin_ray_tune_search_results.log
/mlp/...
/rnn/...
/lstm/...
/transformer/...
/ppo/...
```
