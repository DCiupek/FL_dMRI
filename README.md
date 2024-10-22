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
python DL_train.py  
```

## Federated learning with multiple datasets

