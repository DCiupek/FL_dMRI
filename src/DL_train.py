"""
Author: Dominika Ciupek
"""



import torch
from torch.utils.data import DataLoader, random_split
import pickle
from diffusion_data import ImagMRIDataset
from models import ParamUNet
from train_evaluate import train
import wandb
import os
import argparse
import warnings


# Custom function to validate additional mandatory arguments
def validate_args(args):
    if args.wandb == 'Yes':
        if args.key is None:
            warnings.warn("Weights & Biases key not provided, you need to create a Weights & Biases account, or use an existing account", UserWarning)

# Create the parser
parser = argparse.ArgumentParser(description="Script to train the model using a single dataset.")

# Create the parser
parser = argparse.ArgumentParser(description="Script to train the model using a single dataset.", formatter_class=argparse.RawTextHelpFormatter)

# Add arguments
parser.add_argument('parameter', type=str, help='Microstructural parameter, e.g. FA.')
parser.add_argument('max_value', type=float, help='Microstructural parameter maximum value (necessary for normalization), e.g. 1 for FA.')
parser.add_argument('b', type=float, help='b-value, e.g. 1000')
parser.add_argument('epochs', type=int, help='Number of epochs, e.g. 50.')
parser.add_argument('batch', type=int, help='Batch size, e.g. 16.')
parser.add_argument('learning_rate', type=float, help='Learning rate, e.g. 0.001.')
parser.add_argument('dataset', type=str, help='Dataset name, e.g. CamCan.')
parser.add_argument('save_path', type=str, help='Path to save the model, e.g. C:/Users/.')
parser.add_argument('wandb', type=str, help="Use Weights & Biases for monitoring, Yes or No.\n  - If 'Yes', the following arguments are required: --key.\n")
parser.add_argument('--key', type=str, help='Weights & Biases key.', required=False)

# Parse the arguments
args = parser.parse_args()

# Validate arguments based on wandb
validate_args(args)

# Set global parameters
PARAMETER = args.parameter
MAX_VAL = args.max_value
B_VAL = args.b
NO_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR_RATE = args.learning_rate
DATASET = args.dataset
MODEL_PATH = args.save_path

WANDB_RUN = True if args.wandb == "Yes" else False

IN_SIZE = 30
OUT_SIZE = 1
FEATURES = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ParamUNet_' + PARAMETER + '_' + DATASET + '_L1multDSSIM_normNew_ep-' + str(NO_EPOCHS) + '_batch-' + str(BATCH_SIZE) + '_lr-' + str(LR_RATE)
L = 6

# Set Weights & Biases
if args.wandb == "Yes":

    # Log in to Weights & Biases
    wandb.login(key = args.key)
    
    # Initialize the run
    wandb.init(
        name=MODEL_NAME,
        project="test_" + PARAMETER,   
        config={
        "learning_rate": LR_RATE,
        "batch": BATCH_SIZE,
        "dataset": DATASET,
        "local_epochs": NO_EPOCHS,
        }
    )

# Load file with data paths
with open(os.getcwd() + '/paths/' + DATASET + '_train_paths.pkl', 'rb') as f:
    paths_dict = pickle.load(f)

# Create dataset class
dataset = ImagMRIDataset(DATASET, PARAMETER, paths_dict, MAX_VAL, B_VAL, L)

# Split dataset
train_split = 0.8
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Create ParamUNet model
model = ParamUNet(IN_SIZE, FEATURES, OUT_SIZE).to(DEVICE)

# Train model
_, _ = train(model, train_loader, val_loader, NO_EPOCHS, LR_RATE, DEVICE, MODEL_NAME, MODEL_PATH, WANDB_RUN)
