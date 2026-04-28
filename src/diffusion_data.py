# ruff: noqa: E741, E501, N806, N803, N802, PLR0913, PLR0917, PLR2004
"""
Author: Tomasz Pięciak, Dominika Ciupek
"""
from __future__ import annotations

import contextlib
import functools
import itertools
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

from dipy.io import read_bvals_bvecs
import joblib
import nibabel as nib
import numpy as np
from scipy import linalg, special
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    x = bvecs[:, 0]
    y = bvecs[:, 1]
    z = bvecs[:, 2]

    xyz_sqrt = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # To spherical representation
    phi = np.arccos(np.divide(z, xyz_sqrt, out=np.zeros_like(z), where=(xyz_sqrt != 0)))      # polar angle
    theta = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=(x != 0)))                  # azimuthal angle

    # Correct the azimuthal angle if x < 0
    theta += np.pi * ((x < 0) * 1.0)

    # Generate spherical harmonics basis
    R = int((l + 1) * (l + 2) / 2)
    B = generate_matrix_B(theta, phi, bvecs, l)

    # Laplace-Beltrami operator
    L = generate_matrix_L(l)

    # Inverse solution:
    W_inv = np.matmul(linalg.inv(np.matmul(np.conj(B).T, B) + lambda_reg * L), np.conj(B).T)
    C = np.tensordot(signal, W_inv, axes=([signal.ndim - 1], [1]))
    if C.shape[-1] != R:
        msg = (
            "The last dimension of the SH representation should be equal to R = (l+1)*(l+2)/2.\n"
            f"Got {C.shape[-1]} instead of R={R}."
        )
        raise ValueError(msg)
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

    expected_shape = (*C.shape[0:3], bvecs.shape[0])
    # Convert all gradients from Cartesian to a spherical representation
    # (x, y, z) -> (r, phi, theta)
    x = bvecs[:, 0]
    y = bvecs[:, 1]
    z = bvecs[:, 2]

    xyz_sqrt = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # To spherical representation
    phi = np.arccos(np.divide(z, xyz_sqrt, out=np.zeros_like(z), where=(xyz_sqrt != 0)))      # polar angle
    theta = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=(x != 0)))                  # azimuthal angle

    # Correct the azimuthal angle if x < 0
    theta += np.pi * ((x < 0) * 1.0)

    # Generate spherical harmonics basis
    B = generate_matrix_B(theta, phi, bvecs, l)

    signal = np.einsum('...p,op->...o', C, B)
    if signal.shape != expected_shape:
        msg = (
            f"The shape of the reconstructed signal should be equal to {expected_shape}, "
            f"but got {signal.shape}."
        )
        raise ValueError(msg)
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
    R = int((l + 1) * (l + 2) / 2)
    B = np.zeros([np.max(bvecs.shape), R])

    # Iterate over k = 0, 2, 4, ..., l
    # Iterate over m = -k, ..., 0, ...k
    for k in np.arange(0, l + 1, 2):
        for m in np.arange(-k, k + 1):

            j = int((k**2 + k + 2) / 2 + m)

            # Construct matrix B
            if m == 0:
                B[:, j - 1] = np.real(special.sph_harm(0, k, theta, phi))

            elif -k <= m:
                B[:, j - 1] = np.sqrt(2) * np.real(special.sph_harm(m, k, theta, phi))

            else:
                B[:, j - 1] = np.sqrt(2) * np.imag(special.sph_harm(m, k, theta, phi))

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

    R = int((l + 1) * (l + 2) / 2)
    L = np.zeros([R, R])

    # Iterate over k = 0, 2, 4, ..., l
    # Iterate over m = -k, ..., 0, ...k
    for k in np.arange(0, l + 1, 2):
        for m in np.arange(-k, k + 1):

            j = int((k**2 + k + 2) / 2 + m)
            L[j - 1, j - 1] = np.copy(k)

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

    bvals_new_idx = np.squeeze(np.where((bvals_new >= bvalue - eps) & (bvals_new <= bvalue + eps)))

    bvals_orig_idx = np.squeeze(np.where((bvals_orig >= bvalue - eps) & (bvals_orig <= bvalue + eps)))

    # Transform the diffusion-weighted MR signal into the representation of SH
    data_sh_representation = signal2sh(signal[..., bvals_orig_idx], bvecs_orig[bvals_orig_idx, :], l, lambda_reg)

    print(signal[..., bvals_orig_idx].shape)

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
    signal = np.divide(signal, np.repeat(signal_b0[:, :, :, np.newaxis], signal.shape[3], axis=3))

    signal[signal > 1] = 1
    signal[np.isnan(signal)] = 0
    signal[signal < 0] = 0

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
    par /= max_val
    par[par > 1] = 1
    par[np.isnan(par)] = 0
    par[par < 0] = 0

    return par


def load_data(dataset, parameter, data_path, par_path, bvals_path,
              bvecs_path, max_val, bvalue, l, in_size, data_type, lambda_reg=0.006, eps=100):

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
    signal = nib.load(data_path).get_fdata()
    par = nib.load(str(Path(par_path) / (parameter + '.nii'))).get_fdata()
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)

    # Select diffusion-weighted MR signal for b-value equal to 0
    signal_b0 = signal[:, :, :, np.squeeze(np.where(bvals < eps))]

    # Check needed datatype
    if data_type == "dMRI":

        # Transform the diffusion-weighted MR signal using the selected b-vector
        if in_size in {32, 128, 6}:

            path_new = ''
            assert path_new, "Define the path to the new b-vectors for in_size = 32, 128, 6. Sorry for the inconvenience."

            _, bvecs_new = read_bvals_bvecs(None, path_new + 'samples_' + str(in_size) + '.txt')

            bvals_orig_idx = np.squeeze(np.where((bvals >= bvalue - eps) & (bvals <= bvalue + eps)))

            # Transform the diffusion-weighted MR signal into the representation of SH
            data_sh_representation = signal2sh(signal[..., bvals_orig_idx], bvecs[bvals_orig_idx, :], l, lambda_reg)

            # Transform the representation of SH into the diffusion-weighted MR signal
            signal = sh2signal(data_sh_representation, bvecs_new, l)

        elif in_size == 64:
            if dataset not in {'UCLA', 'UCLA_bipolar', 'UCLA_all', 'UCLA_all_30', 'UCLA_30'}:
                print("Modified")
                print(dataset)

                path_new = '/net/pr2/projects/plgrid/plggflmri/Data/Raw/UCLA/ds000030_R1_05/sub-10159/dwi/'
                bvals_new, bvecs_new = read_bvals_bvecs(path_new + 'sub-10159_dwi.bval', path_new + 'sub-10159_dwi.bvec')

                signal = transform_signal_bvecs(signal, bvals, bvecs, bvals_new, bvecs_new, bvalue, l, lambda_reg, eps)

            else:

                print("Original")
                print(dataset)

                bvals_orig_idx = np.squeeze(np.where((bvals >= bvalue - eps) & (bvals <= bvalue + eps)))
                signal = signal[..., bvals_orig_idx]

        elif in_size == 90:

            if dataset not in {'Zenodo', 'Zenodo_bipolar', 'Zenodo_all'}:

                print("Modified")
                print(dataset)

                path_new = '/net/pr2/projects/plgrid/plggflmri/Data/Raw/zenodo.3935636/'
                bvals_new, bvecs_new = read_bvals_bvecs(path_new + 'crl_132_diff.bval', path_new + 'crl_132_diff.bvec')

                signal = transform_signal_bvecs(signal, bvals, bvecs, bvals_new, bvecs_new, bvalue, l, lambda_reg, eps)

            else:

                print("Original")
                print(dataset)

                bvals_orig_idx = np.squeeze(np.where((bvals >= bvalue - eps) & (bvals <= bvalue + eps)))
                signal = signal[..., bvals_orig_idx]

        elif in_size == 30:

            if dataset != 'Multimodal':

                path_new = '/net/pr2/projects/plgrid/plggflmri/Data/Raw/Multicenter_dMRI/sub-1/ses-c06r1/dwi/'
                bvals_new, bvecs_new = read_bvals_bvecs(path_new + 'sub-1_ses-c06r1_dwi.bval', path_new + 'sub-1_ses-c06r1_dwi.bvec')

                signal = transform_signal_bvecs(signal, bvals, bvecs, bvals_new, bvecs_new, bvalue, l, lambda_reg, eps)

            else:

                bvals_orig_idx = np.squeeze(np.where((bvals >= bvalue - eps) & (bvals <= bvalue + eps)))
                signal = signal[..., bvals_orig_idx]

        # Normalize the diffusion-weighted MR signal
        signal = signal_norm(signal, signal_b0)

    if data_type == "RISH":

        print("Error")

    # Normalize the microstructural parameter
    par = parameters_norm(par, max_val)

    return signal, par


def unzip(iterable: Iterable[tuple], strict=False) -> tuple[Iterable, ...]:
    """
    The inverse of zip object. It yields tuple of iterables, where the i-th iterable
    contains the i-th element from each of the argument sequences or iterables.

    This continues until the first mismatch with the first argument length.
    If strict is true and such situation occurs raise a ValueError.
    """
    it = iter(iterable)
    fst = next(it)
    n = len(fst)
    iterables = itertools.tee(it, n)

    def rest(i):
        for x in iterables[i]:
            if len(x) == n:
                yield x[i]
                continue
            if strict:
                msg = (
                    f"unzip() argument is not a valid zip object, because the element {x}"
                    f" has a different length than {len(fst)}"
                )
                raise ValueError(msg)
            return

    return tuple(itertools.chain((fst[i],), iter(rest(i))) for i in range(n))


def _load_single_mri_image(fil_data, fil_par, fil_bvals, fil_bvecs, dataset,
                           parameter, max_val, bvalue, l, in_size, data_type,
                           lambda_reg, eps):
    mris = []
    parameters = []

    if not (Path(fil_par) / (parameter + '.nii')).exists():
        return (mris, parameters)
    signal, par = load_data(dataset, parameter, fil_data, fil_par, fil_bvals,
                            fil_bvecs, max_val, bvalue, l, in_size, data_type, lambda_reg, eps)
    print(signal.shape)

    # Check if diffusion-weighted MR signal has proper shape
    if signal.shape[3] != in_size:
        return (mris, parameters)

    print('Yes')

    # Move the axis corresponding to the b-vectors/number of RISH
    signal = np.moveaxis(signal, -1, 0)

    a, b, c = [round(signal.shape[i + 1] / 2) for i in range(3)]

    print(c - 20)
    print(c + 20)

    for i in range(c - 20, c + 20):

        # Select slices
        slices = [(slice(a - 40, a + 24), slice(b - 40, b + 24)),
                  (slice(a - 24, a + 40), slice(b - 24, b + 40)),
                  (slice(a - 24, a + 40), slice(b - 40, b + 24)),
                  (slice(a - 40, a + 24), slice(b - 24, b + 40))]

        mris.extend([signal[:, s, r, i] for s, r in slices])
        parameters.extend([par[s, r, i] for s, r in slices])
    return mris, parameters


def _get_node_memory_from_env_variable() -> int | None:
    mem_mb = os.environ["SLURM_MEM_PER_NODE"]
    return int(mem_mb) // 1024


def _get_node_memory_from_scontrol() -> int | None:
    partition = os.environ["SLURM_JOB_PARTITION"]
    cpus = int(os.environ["SLURM_CPUS_ON_NODE"])
    scontrol_cmd = [
        '/net/slurm/releases/production.x86_64/bin/scontrol',
        'show',
        'partition',
        f'{partition}'
    ]
    proc = subprocess.Popen(scontrol_cmd, stdout=subprocess.PIPE)
    output = proc.stdout.read().decode('utf-8')
    mem_line = next(line for line in output.splitlines() if "DefMemPerCPU" in line)
    mem_entry = next(entry for entry in mem_line.strip().split() if entry.startswith("DefMemPerCPU="))
    mem_per_cpu_mb = int(mem_entry.split('=')[1].strip().rstrip('MB'))
    return (mem_per_cpu_mb * cpus) // 1024


def _node_get_memory_gb() -> int | None:
    attempts = [
        _get_node_memory_from_env_variable,
        _get_node_memory_from_scontrol,
    ]

    for attempt in attempts:
        with contextlib.suppress(Exception):
            return attempt()

    return None


def _estimate_number_of_workers() -> int:
    memory_gb = _node_get_memory_gb()
    if memory_gb is None:
        return 1

    # Empirically estimated that each worker requires around 14 GB of memory, with a base
    # overhead of 16 GB for the main process and other system operations. These parameters
    # are based on tests with HCP dataset and may need adjustment for different datasets or
    # system configurations.
    return int(1 + (memory_gb - 16) / 14)


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

    def __init__(self, dataset, parameter, paths_dict, max_val, bvalue, l, in_size, data_type,
                 lambda_reg=0.006, eps=100, transform=None):

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
        in_size: int
            Number of channels in AI model.
        data_type: str
            Data used as input (dMRI or RISH).
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

        # Load paths to files
        data_path = self.paths_dict["data_path"]
        par_path = self.paths_dict["par_path"]
        bvals_path = self.paths_dict["bvals_path"]
        bvecs_path = self.paths_dict["bvecs_path"]

        # HERE
        mw = _estimate_number_of_workers()

        load_single = functools.partial(_load_single_mri_image, dataset=dataset, parameter=parameter, max_val=max_val,
                          bvalue=bvalue, l=l, in_size=in_size, data_type=data_type, lambda_reg=lambda_reg, eps=eps)

        parallel = joblib.Parallel(n_jobs=mw)
        paths = zip(data_path, par_path, bvals_path, bvecs_path, strict=False)
        parallel_gen = (joblib.delayed(load_single)(*ps) for ps in paths)
        self.mri, self.parameters = (np.fromiter(x, dtype=np.float64) for x in unzip(parallel(parallel_gen)))

        # Change list to ndarrays
        self.parameters = self.parameters.reshape((self.parameters.shape[0], 1, 64, 64))

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

    def save_joblib(self, path: str | Path, compress: int = 3) -> None:
        joblib.dump(self, Path(path), compress=compress)

    @classmethod
    def load_joblib(cls, path: str | Path) -> ImagMRIDataset:
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            msg = f"Expected file to contain an object of type {cls.__name__}, but got {type(obj).__name__}"
            raise TypeError(msg)
        return obj


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
