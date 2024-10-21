"""
Author: Tomasz Pięciak, Dominika Ciupek
"""



import numpy as np
from scipy import special
from scipy import linalg
import torch
import os
import nibabel as nib
from torch.utils.data import Dataset
from dipy.io import read_bvals_bvecs
from torchvision.transforms import Compose



def signal2sh(signal, bvecs, l, lambda_reg=0.006):

    """ 
    Transforms the diffusion-weighted MR signal into the representation of spherical harmonics (SHs).
    
    Function uses the notation (r, phi, theta):
      - phi is the polar (inclination) angle, phi in [0, 2pi);
      - theta is the azimuthal angle, theta in [0, pi).                                                      

    
    Parameters:
    ----------
    signal: ndarray
        Diffusion-weighted MR signal of size nxG.
    bvecs: 2darray
        b-vectors matrix of size Gx3.
    l: int
        Order of the SH decomposition.
    lambda_reg: float, optional
        Regularization parameter used for the Laplace-Beltrami matrix. Defaults to 0.006.
    
    Returns:
    -------
    C: ndarray
        SH representation of size nxR. R = (l+1)*(l+2)/2.
    
    """    
    
    # Check if the bvecs vector has more rows than columns
    if bvecs.shape[0] < bvecs.shape[1]:
        bvecs = bvecs.T
    
    # Convert all gradients from Cartesian to a spherical representation
    # (x, y, z) -> (r, phi, theta)
    x = bvecs[:,0]
    y = bvecs[:,1]
    z = bvecs[:,2]
    
    xyz_sqrt = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    # To spherical representation
    phi = np.arccos(np.divide(z, xyz_sqrt, out=np.zeros_like(z), where=(xyz_sqrt!=0)))      # polar angle
    theta = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=(x!=0)))                  # azimuthal angle
    
    # Correct the azimuthal angle if x < 0
    theta = theta + np.pi * ((x < 0) * 1.0)
    
    # Generate spherical harmonics basis
    R = int((l+1)*(l+2)/2)
    B = generate_matrix_B(theta, phi, bvecs, l)
    
    # Laplace-Beltrami operator
    L = generate_matrix_L(l)
   
    # Inverse solution:
    W_inv = np.matmul(linalg.inv(np.matmul(np.conj(B).T, B) + lambda_reg * L), np.conj(B).T)
    

    if signal.ndim == 1:          # 1D signal
        C = W_inv.dot(signal)
        
    elif signal.ndim == 3:        # 2D signal
        C = np.zeros([signal.shape[0], signal.shape[1], R])
        
        for ii in np.arange(signal.shape[0]):
            for jj in np.arange(signal.shape[1]):
                C[ii,jj,:] = W_inv.dot(signal[ii,jj,:])
    
    elif signal.ndim == 4:        # 3D signal
        C = np.zeros([signal.shape[0], signal.shape[1], signal.shape[2], R])
        
        for ii in np.arange(signal.shape[0]):
            for jj in np.arange(signal.shape[1]):
                for kk in np.arange(signal.shape[2]):
                    C[ii,jj,kk,:] = W_inv.dot(signal[ii,jj,kk,:])
            
    return C



def sh2signal(C, bvecs, l):
    
    """ 
    Transforms the representation of spherical harmonics (SHs) into the diffusion-weighted MR signal.
    
    Function uses the notation (r, phi, theta):
      - phi is the polar (inclination) angle, phi in [0, 2pi);
      - theta is the azimuthal angle, theta in [0, pi).                                                      

    
    Parameters:
    ----------
    C: ndarray
        SH representation of size nxR. R = (l+1)*(l+2)/2.
    bvecs: 2darray
        b-vectors matrix of size Gx3.
    l: int
        Order of the SH decomposition.
    
    Returns:
    -------
    signal: ndarray
        Diffusion-weighted MR signal of size nxG.
    
    """    
        
    signal = np.zeros(list(C.shape[0:3]) + [bvecs.shape[0]])
    
    # Convert all gradients from Cartesian to a spherical representation
    # (x, y, z) -> (r, phi, theta)
    x = bvecs[:,0]
    y = bvecs[:,1]
    z = bvecs[:,2]
    
    xyz_sqrt = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    # To spherical representation
    phi = np.arccos(np.divide(z, xyz_sqrt, out=np.zeros_like(z), where=(xyz_sqrt!=0)))      # polar angle
    theta = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=(x!=0)))                  # azimuthal angle
    
    # Correct the azimuthal angle if x < 0
    theta = theta + np.pi * ((x < 0) * 1.0)
    
    # Generate spherical harmonics basis
    B = generate_matrix_B(theta, phi, bvecs, l)
    
    for ii in np.arange(signal.shape[0]):
        for jj in np.arange(signal.shape[1]):
            for kk in np.arange(signal.shape[2]):
                
                signal[ii,jj,kk,:] = np.matmul(B, C[ii,jj,kk,:])

    return signal



def generate_matrix_B(theta, phi, bvecs, l):
    
    """ 
    Generates the matrix B constructed with the discrete modified SH basis.                  

    
    Parameters:
    ----------
    theta: float
        Azimuthal angle in [0, pi).
    phi: float
        Polar (inclination) angle in [0, 2pi).
    bvecs: 2darray
        b-vectors matrix of size Gx3.
    l: int
        Order of the SH decomposition.
    
    Returns:
    -------
    B: 2darray
        Discrete modified SH basis matrix of size GxR. R = (l+1)*(l+2)/2.
    
    """    
    
    # Generate spherical harmonics basis
    R = int((l+1)*(l+2)/2)
    B = np.zeros([np.max(bvecs.shape), R])
    
    # Iterate over k = 0, 2, 4, ..., l
    # Iterate over m = -k, ..., 0, ...k
    for k in np.arange(0, l+1, 2):
        for m in np.arange(-k, k+1):
            
            j = int((k**2 + k + 2)/2 + m)
            
            # Construct matrix B
            if m == 0:
                B[:,j-1] = np.real(special.sph_harm(0, k,  theta, phi))
                
            elif -k <= m:
                B[:,j-1] = np.sqrt(2) * np.real(special.sph_harm(m, k,  theta, phi))
                
            else:
                B[:,j-1] = np.sqrt(2) * np.imag(special.sph_harm(m, k,  theta, phi))
    
    return B
    


def generate_matrix_L(l):
    
    """ 
    Generates the Laplace-Beltrami matrix.     

    
    Parameters:
    ----------
    l: int
        Order of the SH decomposition.
    
    Returns:
    -------
    L: 2darray
        Laplace-Beltrami matrix of size RxR. R = (l+1)*(l+2)/2.
    
    """    
    
    R = int((l+1)*(l+2)/2)
    L = np.zeros([R, R])
    
    # Iterate over k = 0, 2, 4, ..., l
    # Iterate over m = -k, ..., 0, ...k
    for k in np.arange(0, l+1, 2):
        for m in np.arange(-k, k+1):
            
            j = int((k**2 + k + 2)/2 + m)
            L[j-1,j-1] = np.copy(k)
    
    return L



def transform_signal_bvecs(signal, bvals_orig, bvecs_orig, bvals_new, 
                         bvecs_new, bvalue, l, lambda_reg=0.006, eps=100):
    
    """ 
    Transforms the initial diffusion-weighted MR signal for defined b-value using a provided b-vector.    

    
    Parameters:
    ----------
    signal: ndarray
        Diffusion-weighted MR signal of size nxG.
    bvals_orig: 1darray
        Initial b-values matrix of size Gx1.
    bvecs_orig: 2darray
        Initial b-vectors matrix of size Gx3.
    bvals_new: 1darray
        Target b-values matrix of size Hx1.
    bvecs_new: 2darray
        Target b-vectors matrix of size Hx3.
    bvalue: int
        Defined b-value.
    l: int
        Order of the SH decomposition.
    lambda_reg: float, optional
        Regularization parameter used for the Laplace-Beltrami matrix. Defaults to 0.006.
    eps: int, optional
        Uncertainty value for the defined b-value. Defaults to 100.
            
    Returns:
    -------
    signal_reconstructed: ndarray
        Modified diffusion-weighted MR signal of size nxH.
    
    """    
    
    bvals_new_idx = np.squeeze(np.where((bvals_new >= bvalue-eps) & (bvals_new <= bvalue+eps)))
         
    bvals_orig_idx = np.squeeze(np.where((bvals_orig >= bvalue-eps) & (bvals_orig <= bvalue+eps)))
    
    # Transform the diffusion-weighted MR signal into the representation of SH
    data_sh_representation = signal2sh(signal[..., bvals_orig_idx], bvecs_orig[bvals_orig_idx, :], l, lambda_reg) 
    
    # Transform the representation of SH into the diffusion-weighted MR signal
    signal_reconstructed = sh2signal(data_sh_representation, bvecs_new[bvals_new_idx], l)
     
    return signal_reconstructed



def signal_norm(signal, signal_b0):
    
    """ 
    Normalizes the diffusion-weighted MR signal.

    
    Parameters:
    ----------
    signal: ndarray
        Diffusion-weighted MR signal of size nxG.
    signal_b0: ndarray
        Diffusion-weighted MR signal for b-value equal to 0 of size nxH.
        
    Returns:
    -------
    signal: ndarray
        Normalized diffusion-weighted MR signal.
    
    """    
    
    # Calculate the mean value of diffusion-weighted MR signal for b-value equal to 0
    if signal_b0.ndim == 4:
        signal_b0 = np.mean(signal_b0, axis=3)
    
    # Divide the diffusion-weighted MR signal by the diffusion-weighted MR signal for b-value equal to 0
    signal = np.divide(signal, np.repeat(signal_b0[:,:,:, np.newaxis], signal.shape[3], axis=3))
    
    signal[signal>1] = 1
    signal[np.isnan(signal)] = 0
    signal[signal<0] = 0
    
    return signal
    
    
    
def parameters_norm(par, max_val):
    
    """ 
    Normalizes the values of a microstructural parameter.

    
    Parameters:
    ----------
    par: ndarray
        Microstructural parameter.
    max_val: float
        Maximum biological value of a microstructural parameter.
        
    Returns:
    -------
    par: ndarray
        Normalized microstructural parameter.
    
    """    
    
    # Divide the microstructural parameter by the maximum biological value
    par = par/max_val
    par[par>1] = 1
    par[np.isnan(par)] = 0
    par[par<0] = 0
    
    return par
    
    

def load_data(dataset, parameter, data_path, par_path, bvals_path, 
              bvecs_path, max_val, bvalue, l, lambda_reg=0.006, eps=100):
    
    """ 
    Loads Diffusion-weighted MR signal and microstructural parameter.

    
    Parameters:
    ----------
    dataset: str
        Name of the dataset.
    parameter: str
        Name of the microstructural parameter.
    data_path: str
        Path to the diffusion-weighted MR signal.
    par_path: str
        Path to the microstructural parameter.
    bvals_path: str 
        Path to the b-values matrix.
    bvecs_path: str
        Path to the b-vectors matrix.
    max_val: float
        Maximum biological value of a microstructural parameter.
    bvalue: int
        Defined b-value.
    l: int
        Order of the SH decomposition.
    lambda_reg: float, optional
        Regularization parameter used for the Laplace-Beltrami matrix. Defaults to 0.006.
    eps: int, optional
        Uncertainty value for the defined b-value. Defaults to 100.
        
    Returns:
    -------
    signal: ndarray
        Normalized single diffusion-weighted MR signal.
    par: ndarray
        Normalized single microstructural parameter.
    
    """
    
    # Load diffusion-weighted MR signal and microstructural parameter
    signal = nib.load(data_path).get_data()
    par = nib.load(par_path + parameter + '.nii').get_data()
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
    
    # Select diffusion-weighted MR signal for b-value equal to 0
    signal_b0 = signal[:, :, :, np.squeeze(np.where(bvals < eps))]
    
    # Transform the diffusion-weighted MR signal using the b-vector for Multimodal dataset    
    if dataset != 'Multimodal':

        path_new = '/net/pr2/projects/plgrid/plggflmri/Data/Raw/Multicenter_dMRI/sub-1/ses-c06r1/dwi/'        
        bvals_new, bvecs_new = read_bvals_bvecs(path_new + 'sub-1_ses-c06r1_dwi.bval', path_new + 'sub-1_ses-c06r1_dwi.bvec')
        
        signal = transform_signal_bvecs(signal, bvals, bvecs, bvals_new, bvecs_new, bvalue, l, lambda_reg, eps)    
        
    else:

        bvals_orig_idx = np.squeeze(np.where((bvals >= bvalue-eps) & (bvals <= bvalue+eps)))
        signal = signal[..., bvals_orig_idx]
         
    
    # Normalize the diffusion-weighted MR signal and the microstructural parameter
    signal = signal_norm(signal, signal_b0)
    par = parameters_norm(par, max_val)
    
    return signal, par


class ImagMRIDataset(Dataset):

    """ 
    A class used to represent full MRI dataset.
    
    
    Attributes:
    ----------
    dataset: str
        Name of the dataset.
    parameter: str
        Name of the microstructural parameter.
    paths_dict: dict
        Dictionary containing the paths to the files of the given dataset:
            data_path, paths to the diffusion-weighted MR signals;
            par_path, paths to the microstructural parameters;
            bvals_path, paths to the b-values matrix;
            bvecs_path, paths to the b-vectors matrix.
    max_val: float
        Maximum biological value of a microstructural parameter.
    bvalue: int
        Defined b-value.
    l: int
        Order of the SH decomposition.
    lambda_reg: float, optional
        Regularization parameter used for the Laplace-Beltrami matrix. Defaults to 0.006.
    eps: int, optional
        Uncertainty value for the defined b-value. Defaults to 100.
    transform: Compose
        Sequence of transformations applied to the original data.
    
    Methods:
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(index)
        Returns a single element at a given index from the dataset.
    
    """    

    def __init__(self, dataset, parameter, paths_dict, max_val, bvalue, l, 
                 lambda_reg=0.006, eps=100, transform = None):
                 
        """ 
        Class constructor.

        
        Parameters:
        ----------
        dataset: str
            Name of the dataset.
        parameter: str
            Name of the microstructural parameter.
        paths_dict: dict
            Dictionary containing the paths to the files of the given dataset:
                data_path, paths to the diffusion-weighted MR signals;
                par_path, paths to the microstructural parameters;
                bvals_path, paths to the b-values matrix;
                bvecs_path, paths to the b-vectors matrix.
        max_val: float
            Maximum biological value of a microstructural parameter.
        bvalue: int
            Defined b-value.
        l: int
            Order of the SH decomposition.
        lambda_reg: float, optional
            Regularization parameter used for the Laplace-Beltrami matrix. Defaults to 0.006.
        eps: int, optional
            Uncertainty value for the defined b-value. Defaults to 100.
        transform: Compose
            Sequence of transformations applied to the original data.
            
        """  
        
        self.dataset = dataset
        self.paths_dict = paths_dict
        self.transform = transform

        self.mri, self.parameters = [], []
        
        # Load paths to files
        data_path = self.paths_dict["data_path"]
        par_path = self.paths_dict["par_path"]
        bvals_path = self.paths_dict["bvals_path"]
        bvecs_path = self.paths_dict["bvecs_path"]
        
        # Iterate over files
        for fil_data, fil_par, fil_bvals, fil_bvecs in zip(data_path, par_path, bvals_path, bvecs_path):
            
            # Check if file with microstructural parameter exist
            if os.path.isfile(fil_par + parameter + '.nii'):
                
                # Load diffusion-weighted MR signal and microstructural parameter
                signal, par = load_data(dataset, parameter, fil_data, fil_par, fil_bvals, 
                                        fil_bvecs, max_val, bvalue, l, lambda_reg, eps)
                                        
                print(signal.shape)
                
                # Check if diffusion-weighted MR signal has proper shape
                if signal.shape[3] == 30:
                
                    print('Yes')
                    
                    # Move the axis corresponding to the b-vectors
                    signal = np.moveaxis(signal, -1, 0)
                    
                    a, b, c = [round(signal.shape[i+1]/2) for i in range(3)]
                
                    for i in range(c-20,c+20):
                        
                        # Select slices
                        slices = [(slice(a-40, a+24), slice(b-40, b+24)), 
                                  (slice(a-24, a+40), slice(b-24, b+40)), 
                                  (slice(a-24, a+40), slice(b-40, b+24)), 
                                  (slice(a-40, a+24), slice(b-24, b+40))]
                        
                        self.mri.extend([signal[:, s, r, i] for s, r in slices])
                        self.parameters.extend([par[s, r, i] for s, r in slices])
        
        # Change list to ndarrays
        self.mri = np.array(self.mri)
        self.parameters = np.array(self.parameters)
        self.parameters = self.parameters.reshape((self.parameters.shape[0],1,64,64))
        
        print(self.parameters.shape)
                    
    def __len__(self):
    
        """ 
        Returns the length of the dataset.

        
        Returns:
        -------
        len(self.mri): int
            Length of the dataset.
        
        """  
        
        return len(self.mri)

    def __getitem__(self, index):
    
        """ 
        Returns a single element at a given index from the dataset.

        
        Parameters:
        ----------
        index: int
            Given index value.
            
        Returns:
        -------
        mri: FloatTensor
            Diffusion-weighted MR signal at a given index.
        parameters: FloatTensor
            Microstructural parameter at a given index.
            
        """  
    
        mri = torch.tensor(self.mri[index]).float()
        parameters = torch.tensor(self.parameters[index]).float()
        
        if self.transform is not None:
            mri = self.transform(mri)
            parameters = self.transform(parameters)

        return mri, parameters
        


def crop_images(shapes, depth):
    
    """ 
    Determines the data dimensions required by the model.

    
    Parameters:
    ----------
    shapes: list
        Shapes of the original data.
    depth: int
        Value by which the shapes must be divisible for the data to be usable by UNet.
        
    Returns:
    -------
    slices: list
        List of slices that must be applied to the data.
        
    """  
    
    slices = []
    
    for shape in shapes:
        rest = shape % depth
        start = round(rest / 2) if rest % 2 == 0 else round(rest / 2) + 1
        slices.append(slice(start, shape - round(rest / 2)))
        
    return slices