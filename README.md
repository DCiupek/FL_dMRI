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

After training the model, it is necessary to rename the best one used for a full evaluation to: `FAUNet_[microstructural parameter]_[dataset name]_L1multDSSIM_normNew_final.pt`.

Run the model evaluation algorithm:
```shell
cd src
python DL_evaluate.py [-h] [--parameters_path PARAMETERS_PATH] [--metrics_path METRICS_PATH] parameter max_value b dataset dataset_train model_path save_parameters save_metrics
```

Args:
```text
positional arguments:
  parameter             Microstructural parameter, e.g. FA.
  max_value             Microstructural parameter maximum value (necessary for normalization), e.g. 1 for FA.
  b                     b-value, e.g. 1000
  dataset               Dataset name, e.g. CamCan.
  dataset_train         Name of the dataset on which the model was trained, e.g. CamCan.
  model_path            Path to the saved model, e.g. C:/Users/.
  save_parameters       Save estimated microstructural parameters, Yes or No.
  save_metrics          Save calculated metrics, Yes or No.

optional arguments:
  -h, --help            show this help message and exit
  --parameters_path PARAMETERS_PATH
                        Path to save the estimated microstructural parameters, required if you want to save estimated
                        microstructural parameters, e.g. C:/Users/.
  --metrics_path METRICS_PATH
                        Path to save the calculated metrics, required if you want to save calculated metrics, e.g.
                        C:/Users/.
```

#### Example of the model evaluation that includes saving estimated parameters and calculated metrics to files:
```shell
python DL_evaluate.py FA 1 1000 Penthera CamCan C:/Users/ Yes Yes --parameters_path C:/Users/Parameters/ --metrics_path C:/Users/Metrics/
```

#### Example of the model evaluation that does not include saving estimated parameters and calculated metrics to files:
```shell
python DL_evaluate.py FA 1 1000 Penthera CamCan C:/Users/ No No
```

### 3. Calculate additional metrics on estimated microstructural parameters:

## Federated learning with multiple datasets

## TO DO:
- Add functionality to select model input channels.
- Include the option to select b vectors for transforming diffusion-weighted MR data.
- Implement model training on averaged diffusion-weighted MR data.
