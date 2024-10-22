# Federated learning for microstructural parameters estimation based on diffusion-weighted MR data

## Deep learning with a single dataset

### 1. Prepare dataset:

Save the paths to the training and test data in two separate files named `[dataset name]_train_paths.pkl` and `[dataset name]_test_paths.pkl`, located in the `src/paths/` folder.

The saved files should contain a Python dictionary with the following structure:
- `data_path`: a list of paths to the diffusion-weighted MR signals,
- `par_path`: a list of paths to the microstructural parameters,
- `bvals_path`: a list of paths to the b-values matrix,
- `bvecs_path`: a list of paths to the b-vectors matrix.

### 2. Train the model:

Run the model training algorithm:
```shell
cd src
python DL_train.py [-h] [--key KEY] parameter max_value b epochs batch learning_rate dataset save_path wandb
```

Args:
```text
positional arguments:
  parameter      Microstructural parameter, e.g. FA.
  max_value      Microstructural parameter maximum value (necessary for normalization), e.g. 1 for FA.
  b              b-value, e.g. 1000
  epochs         Number of epochs, e.g. 50.
  batch          Batch size, e.g. 16.
  learning_rate  Learning rate, e.g. 0.001.
  dataset        Dataset name, e.g. CamCan.
  save_path      Path to save the model, e.g. C:/Users/.
  wandb          Use Weights & Biases for monitoring, Yes or No.

optional arguments:
  -h, --help     show this help message and exit
  --key KEY      Weights & Biases key, required if you want to use Weights & Biases.
```

#### Example of training the model without using Weights & Biases for real-time monitoring:
```shell
python DL_train.py FA 1 1000 50 16 0.001 CamCan C:/Users/ No
```

#### Example of training the model using Weights & Biases for real-time monitoring:
```shell
python DL_train.py FA 1 1000 50 16 0.001 CamCan C:/Users/ Yes --key 000000
```
The provided key is a non-functional example. To monitor training, you need to enter your private key, which can be found on the Weights & Biases page.

### 2. Evaluate the model:

## Federated learning with multiple datasets

