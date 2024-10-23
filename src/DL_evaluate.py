"""
Author: Dominika Ciupek
"""



import torch
import os
import numpy as np
import nibabel as nib
import pickle
from models import ParamUNet
from diffusion_data import load_data, crop_images
from train_evaluate import predict
from metrics import MSSIM, MSE, visualize
import argparse
import warnings


# Create the parser
parser = argparse.ArgumentParser(description="Script to train the model using a single dataset.")

# Add arguments
parser.add_argument('parameter', type=str, help='Microstructural parameter, e.g. FA.')
parser.add_argument('max_value', type=float, help='Microstructural parameter maximum value (necessary for normalization), e.g. 1 for FA.')
parser.add_argument('b', type=float, help='b-value, e.g. 1000')
parser.add_argument('dataset', type=str, help='Dataset name, e.g. CamCan.')
parser.add_argument('dataset_train', type=str, help='Name of the dataset on which the model was trained, e.g. CamCan.')
parser.add_argument('model_path', type=str, help='Path to the saved model, e.g. C:/Users/.')
parser.add_argument('save_parameters', type=str, help='Save estimated microstructural parameters, Yes or No.')
parser.add_argument('--parameters_path', type=str, help='Path to save the estimated microstructural parameters, required if you want to save estimated microstructural parameters e.g. C:/Users/.', required=False)
parser.add_argument('save_metrics', type=str, help='Save calculated metrics, Yes or No.')
parser.add_argument('--metrics_path', type=str, help='Path to save the calculated metrics, required if you want to save calculated metrics, e.g. C:/Users/.', required=False)

# Parse the arguments
args = parser.parse_args()

# Set global parameters
PARAMETER = args.parameter
MAX_VAL = args.max_value
B_VAL = args.b
DATASET = args.dataset
DATASET_TRAIN = args.dataset_train
MODEL_PATH = args.model_path

IN_SIZE = 30
OUT_SIZE = 1
FEATURES = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'FAUNet_' + PARAMETER + '_' + DATASET_TRAIN + '_L1multDSSIM_normNew_final'
L = 6

# Define model
model = ParamUNet(IN_SIZE, FEATURES, OUT_SIZE).to(DEVICE)
model.double()
model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME + '.pt', map_location = torch.device(DEVICE)))

# Load file with data paths
with open(os.getcwd() + '/paths/' + DATASET + '_test_paths.pkl', 'rb') as f:
    paths_dict = pickle.load(f)
    
# Load paths to files
data_path = paths_dict["data_path"]
par_path = paths_dict["par_path"]
bvals_path = paths_dict["bvals_path"]
bvecs_path = paths_dict["bvecs_path"]

# Define metrices
MSEs, MSSIMs = [], []

# Iterate over files
for fil_data, fil_par, fil_bvals, fil_bvecs in zip(data_path, par_path, bvals_path, bvecs_path):
    
    # Check if file with microstructural parameter exist
    if os.path.isfile(fil_par + PARAMETER + '.nii'):
        
        #Load affine matrix
        affine = nib.load(fil_par + PARAMETER + '.nii').affine
        
        # Load diffusion-weighted MR signal and microstructural parameter
        signal, par = load_data(DATASET, PARAMETER, fil_data, fil_par, fil_bvals, 
                                fil_bvecs, MAX_VAL, B_VAL, L)
        
        # Check if diffusion-weighted MR signal has proper shape
        if signal.shape[3] == 30:
            
            # Generate brain mask
            mask = np.any(signal, axis=-1).astype(int)
            
            # Move the axis corresponding to the b-vectors
            signal = np.moveaxis(signal, -1, 0)
            
            # Construct testing set from single subject
            signal = np.moveaxis(signal, -1, 0)
            par = np.moveaxis(par, -1, 0)
            par = par.reshape((par.shape[0],1,par.shape[1],par.shape[2]))
            
            # Check whether the shape is compatible with the model
            slices = crop_images([par.shape[2], par.shape[3]], 8)
            signal = signal[:,:,slices[0],slices[1]]
            par = par[:,:,slices[0],slices[1]]
            
            # Change data type
            mri = torch.tensor(signal).double()
                        
            # Predict microstructural parameter
            parameters_estimate = predict(model, mri, DEVICE)
            parameters_estimate = parameters_estimate.data.cpu().numpy()
            
            # Change shape of microstrucural parameter
            par = np.squeeze(par)
            parameters_estimate = np.squeeze(parameters_estimate)
            
            # Transform back to original shape
            parameters_estimate = np.moveaxis(parameters_estimate, 0, -1)
            par = np.moveaxis(par, 0, -1)
            
            mask = mask[slices[0],slices[1],:]
            
            # Calculate MSSIM
            mssim = MSSIM(par, parameters_estimate, mask)            
                    
            MSSIMs.append(mssim)
            
            print("MSSIM = ", mssim)
                       
            # Calculate MSE
            mse = MSE(par, parameters_estimate, mask)            
                    
            MSEs.append(mse)
            
            print("MSE = ", mse)
            
            if args.save_parameters == "Yes":
            
                dirname, fname = os.path.split(fil_par)
                
                dirname = dirname.split("DTI/Masked/")[1]
            
                # Check if path is provided
                if args.parameters_path==None:
                    warnings.warn("Path to save the estimated microstructural parameters not provided, parameters will be saved in current directory", UserWarning)
                    path_out = os.getcwd() +'/' + dirname
                else:
                    path_out = args.parameters_path + dirname
                
                # Save predicted data
                path_out = args.parameters_path + dirname
                
                if not os.path.isdir(path_out):
                    os.makedirs(path_out)
    
                nifti_file = nib.Nifti1Image(parameters_estimate, affine=affine)
                nib.save(nifti_file, path_out + '/' + PARAMETER + '_' + MODEL_NAME + '.nii')


if args.save_metrics == "Yes":

    # Check if path is provided
    if args.metrics_path==None:
        warnings.warn("Path to save the calculated metrics not provided, metrics will be saved in current directory", UserWarning)
        path_out = os.getcwd() +'/'
    else:
        path_out = args.metrics_path

    # Save metrics
    with open(path_out + "ssim_" + DATASET + "_" + MODEL_NAME + ".pkl", "wb") as fp:
        pickle.dump(sum(MSSIMs)/len(MSSIMs), fp)
    
    with open(path_out + "mse_" + DATASET + "_" + MODEL_NAME + ".pkl", "wb") as fp:
        pickle.dump(sum(MSEs)/len(MSEs), fp)
    
    # Save a sample slice of the estimated data
    slice_path = path_out + DATASET + '_' + MODEL_NAME
    visualize(par, parameters_estimate, slice_path)