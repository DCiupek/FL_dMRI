"""
Author: Dominika Ciupek
"""



import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import pickle
import wandb



def loss_dssim_l1(output, target, device):
    
    """ 
    Determines the value of a personalized loss function: L1*DSSIM.                           

    
    Parameters:
    ----------
    output: ndarray
        Value of the microstructural parameter estimated by the model.
    target: ndarray
        Actual value of the microstructural parameter.
    device: str, 'cuda' or 'cpu'
        Device on which the data is allocated.
    
    Returns:
    -------
    loss_all: float
        Total value of the loss function.
    loss_DSSIM: float
        Value of the DSSIM component of the loss function.
    loss_L1Loss: float
        Value of the L1 component of the loss function.
    
    """    
    
    # Calculate DSSIM
    SSIM = StructuralSimilarityIndexMeasure(data_range=1, k1=0.01, k2=0.03).to(device)
    loss_DSSIM = (1-SSIM(output, target))/2
    
    # Calculate L1Loss
    L1Loss = torch.nn.L1Loss()
    loss_L1Loss = L1Loss(output, target)

    # Calculate entire loss function
    loss_all = loss_L1Loss * loss_DSSIM
    
    return loss_all, loss_DSSIM, loss_L1Loss

        

def train_epoch(model, trainloader, optimizer, device):
    
    """ 
    Trains the model for a single epoch.                           

    
    Parameters:
    ----------
    model: ParamUNet
        UNet for microstructural parameters estimation.
    trainloader: DataLoader
        Train set.
    optimizer: Optimizer
        Optimization algorithm.
    device: str, 'cuda' or 'cpu'
        Device on which the model and data are allocated.
    
    Returns:
    -------
    train_loss: float
        Total value of the loss function.
    train_DSSIM: float
        Value of the DSSIM component of the loss function.
    train_L1Loss: float
        Value of the L1 component of the loss function.
    
    """    
    
    train_loss = 0.0
    train_DSSIM = 0.0
    train_L1Loss = 0.0
    
    # Set the model in training mode
    model.train()
    
    # Loop over the train set
    for i, data in enumerate(trainloader):
        
        # Send the input to the device
        mri, par = data
        mri, par = mri.to(device), par.to(device)
        
        # Zero out accumulated gradients
        optimizer.zero_grad()
        
        # Perform a forward pass and calculate the training loss
        output = model(mri)
        loss_all, loss_DSSIM, loss_L1Loss = loss_dssim_l1(output, par, device)
        
        # Perform backpropagation
        loss_all.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Add the loss to the total training loss
        train_loss += loss_all.item()*mri.size(0)
        train_DSSIM += loss_DSSIM.item()*mri.size(0)
        train_L1Loss += loss_L1Loss.item()*mri.size(0)
        
    train_loss = train_loss/len(trainloader.sampler)
    train_DSSIM = train_DSSIM/len(trainloader.sampler)
    train_L1Loss = train_L1Loss/len(trainloader.sampler)

    return train_loss, train_DSSIM, train_L1Loss



def val_epoch(model, valloader, device):
    
    """ 
    Evaluates the model for a single epoch.                           

    
    Parameters:
    ----------
    model: ParamUNet
        UNet for microstructural parameters estimation.
    valloader: DataLoader
        Validation set.
    device: str, 'cuda' or 'cpu'
        Device on which the model and data are allocated.
    
    Returns:
    -------
    val_loss: float
        Total value of the loss function.
    val_DSSIM: float
        Value of the DSSIM component of the loss function.
    val_L1Loss: float
        Value of the L1 component of the loss function.
    
    """    
    
    val_loss = 0.0
    val_DSSIM = 0.0
    val_L1Loss = 0.0
    
    # Set the model in evaluation mode
    model.eval()
    
    # Loop over the validation set
    for i, data in enumerate(valloader):
        
        # Send the input to the device
        mri, par = data
        mri, par = mri.to(device), par.to(device)
        
        # Make the predictions and calculate the validation loss
        output = model(mri)        
        loss_all, loss_DSSIM, loss_L1Loss = loss_dssim_l1(output, par, device)
        
        # Add the loss to the total validation loss
        val_loss += loss_all.item()*mri.size(0)
        val_DSSIM += loss_DSSIM.item()*mri.size(0)
        val_L1Loss += loss_L1Loss.item()*mri.size(0)
        
    val_loss = val_loss/len(valloader.sampler)
    val_DSSIM = val_DSSIM/len(valloader.sampler)
    val_L1Loss = val_L1Loss/len(valloader.sampler)

    return val_loss, val_DSSIM, val_L1Loss

    
    
def train(model, trainloader, valloader, no_epochs, lr_rate, device, 
          model_name = None, model_path = None, wandb_log = False, save = True):
    
    """ 
    Performs full training of the model.                           

    
    Parameters:
    ----------
    model: ParamUNet
        UNet for microstructural parameters estimation.
    trainloader: DataLoader
        Train set.
    valloader: DataLoader
        Validation set.
    no_epochs: int
        Number of epochs.
    lr_rate: float
        Learning rate value.
    device: str, 'cuda' or 'cpu'
        Device on which the model and data are allocated.
    model_name: str
        Name under which the model and training history will be saved.
    model_path: str
        The path where the model and training history will be saved.
    wandb_log: bool
        If True the training will be saved in the Weights & Biases service
    
    """
    
    # Define optimization algorithm
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr_rate)
    
    history = {'train_loss': [], 'val_loss': []}
    
    best_train_loss = 100
    best_val_loss = 100
    best_epoch = 0
    
    val_loss_epochs = []
    
    # Loop over epochs
    for epoch in range(no_epochs):
        
        print('Epoch {}'.format(epoch + 1))
        
        # Perform training and evaluation for a single epoch
        train_loss, train_DSSIM, train_L1Loss = train_epoch(model, trainloader, optimizer, device)
        val_loss, val_DSSIM, val_L1Loss = val_epoch(model, valloader, device)
        val_loss_epochs.append(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print('Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training DSSIM:{:.3f} AVG Training RMSE:{:.3f} AVG Validation Loss:{:.3f} AVG Validation DSSIM:{:.3f} AVG Validation RMSE:{:.3f}'.format(epoch + 1, no_epochs, train_loss, train_DSSIM, train_L1Loss, val_loss, val_DSSIM, val_L1Loss))
        
        # Save losses in the Weights & Biases service
        if wandb_log == True:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        # Save best model    
        if save == True:   
            if train_loss <= best_train_loss and val_loss <= best_val_loss:
                
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_epoch = epoch
                
                torch.save(model.state_dict(), model_path + model_name + '.pt')
    
    # Save full training history
    if save == True: 
        with open(model_path + model_name + '_history.pkl', 'wb') as f:
            pickle.dump(history, f)
            
        with open(model_path + model_name + '_best_epoch.pkl', 'wb') as f:
            pickle.dump(best_epoch, f)
            
    return np.mean(val_loss_epochs), val_loss



def predict(model, signal, device):
    
    """ 
    Estimates microstructural parameters.                           

    
    Parameters:
    ----------
    model: ParamUNet
        UNet for microstructural parameters estimation.
    signal: ndarray
        Diffusion-weighted MR signal.        
    device: str, 'cuda' or 'cpu'
        Device on which the model and data are allocated.
    
    Returns:
    -------
    par: ndarray
        Estimated microstructural parameter.
    
    """ 
    
    signal = torch.tensor(signal)
    
    # Send the model and input to the device
    model.to(device)   
    signal = signal.to(device)
    
    # Set the model in evaluation mode
    model.eval()
    
    # Make the predictions
    par = model(signal)
    
    return par
    