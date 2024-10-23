"""
Author: Dominika Ciupek
"""



import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def MSSIM(par, parameters_estimate, mask):

    """ 
    Calculates the Mean Structural Similarity Index Measure (MSSIM).                                                   

    
    Parameters:
    ----------
    par: ndarray
        Microstructural parameter.
    parameters_estimate: ndarray
        Estimated microstructural parameter.
    Mask: ndarray
        Brain or ROI mask.
    
    Returns:
    -------
    mssim: float
        MSSIM value. 
    
    """    
               
    ssim_sum = 0
    num = 0
    
    # Calculate SSIM for a single slice
    for i in range(par.shape[0]):
        ssim_temp = ssim(parameters_estimate[i,:,:], par[i,:,:], ~np.array(mask[i,:,:], dtype=bool), data_range=np.max(parameters_estimate[i,:,:]) - np.min(parameters_estimate[i,:,:]))
        
        # Check if SSIM was calculated correctly
        if isinstance(ssim_temp, np.float64):
            ssim_sum += ssim_temp
            num += 1
    
    # Calculate MSSIM
    mssim = ssim_sum/num
    
    return mssim


def MSE(par, parameters_estimate, mask):

    """ 
    Calculates the Mean Square Error (MSE).                                                   

    
    Parameters:
    ----------
    par: ndarray
        Microstructural parameter.
    parameters_estimate: ndarray
        Estimated microstructural parameter.
    mask: ndarray
        Brain or ROI mask.
    
    Returns:
    -------
    mse: float
        MSE value. 
    
    """   
    
    # Calculate MSE for each voxel               
    mse_list = [(parameters_estimate[x,y,z] - par[x,y,z])**2 for x in range(mask.shape[0]) 
                for y in range(mask.shape[1]) for z in range(mask.shape[2]) if mask[x,y,z] != 0]
    
    # Calculate MSE
    mse = sum(mse_list)/len(mse_list)
    
    return mse
    

def visualize(par, parameters_estimate, save_path):

    """ 
    Visualize example slice of estimated microstructural parameter.                                                   

    
    Parameters:
    ----------
    par: ndarray
        Microstructural parameter.
    parameters_estimate: ndarray
        Estimated microstructural parameter.
    save_path: str
        Path to save the example slice with filename.
    
    """ 

    a = round(par.shape[2]/2)
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.title('Target')
    plt.imshow(np.rot90(par[:,:,a-15],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,2)
    plt.title('Predicted')
    plt.imshow(np.rot90(parameters_estimate[:,:,a-15],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,3)
    plt.title('Error')
    plt.imshow(np.rot90(abs(par[:,:,a-15]-parameters_estimate[:,:,a-15]),-1), cmap='hot', vmin=0, vmax=0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,4)
    plt.title('Target')
    plt.imshow(np.rot90(par[:,:,a],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,5)
    plt.title('Predicted')
    plt.imshow(np.rot90(parameters_estimate[:,:,a],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,6)
    plt.title('Error')
    plt.imshow(np.rot90(abs(par[:,:,a]-parameters_estimate[:,:,a]),-1), cmap='hot', vmin=0, vmax=0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,7)
    plt.title('Target')
    plt.imshow(np.rot90(par[:,:,a+15],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,8)
    plt.title('Predicted')
    plt.imshow(np.rot90(parameters_estimate[:,:,a+15],-1), cmap='hot', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3,3,9)
    plt.title('Error')
    plt.imshow(np.rot90(abs(par[:,:,a+15]-parameters_estimate[:,:,a+15]),-1), cmap='hot', vmin=0, vmax=0.1)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_path + '.png', dpi=300)