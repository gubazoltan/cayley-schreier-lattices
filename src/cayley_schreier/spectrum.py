"""
Due to the fact that the Hamiltonians are constructed with sympy matrices 
and then converted into lambda functions for numerical evaluation,
this module cannot be vectorized easily over k-points.
"""

import numpy as np
from typing import Callable, Sequence, Tuple


def spectrum(
    hamiltonian_function: Callable[..., np.ndarray],
    Brillouin_zone: Sequence[np.ndarray],
    num_bands: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy bands over a k-grid (1D, 2D, or 3D).

    hamiltonian_function:
        For 1D: f(k1: float) -> (M,M) ndarray (Hermitian).
        For 2D: f(k1: float, k2: float) -> (M,M).
        For 3D: f(k1: float, k2: float, k3: float) -> (M,M).
    Brillouin_zone:
        Sequence of 1D arrays; len = dimension. Example (1D): [k_array]
    num_bands:
        Number of bands (must be = matrix dimension).
    Returns:
        eigenvalues  shape: (N1[, N2[, N3]], num_bands)
        eigenvectors shape: (N1[, N2[, N3]], num_bands, num_bands)
    """

    # determine the lattice dimension
    dim = len(Brillouin_zone)

    # quick dimension check at k = 0
    if hamiltonian_function(*([0 for _ in range(dim)])).shape[0] != num_bands:
        raise ValueError("Hamiltonian matrix dimension must match num_bands.")
    
    # dispatch to the appropriate routine
    match dim: 
        case 1:
            return _spectrum_1d(hamiltonian_function, Brillouin_zone, num_bands)
        case 2:
            return _spectrum_2d(hamiltonian_function, Brillouin_zone, num_bands)
        case 3:
            return _spectrum_3d(hamiltonian_function, Brillouin_zone, num_bands)
        case _:
            raise ValueError("Brillouin zone dimension must be 1, 2, or 3.")

def _spectrum_1d(
    hamiltonian_function: Callable[[float], np.ndarray],
    Brillouin_zone: Sequence[np.ndarray],
    num_bands: int
):
    #unpack BZ points
    K1s = Brillouin_zone[0] 
    N1 = len(K1s)

    #initialize arrays to hold eigenvalues and eigenvectors
    eigenvalues = np.zeros((N1, num_bands), dtype=np.float64)
    eigenvectors = np.zeros((N1, num_bands, num_bands), dtype=np.complex128)

    for i1, k1 in enumerate(K1s):
        vals, vecs = np.linalg.eigh(hamiltonian_function(k1))

        ind = np.argsort(vals)  #sort eigenvalues and eigenvectors
        eigenvalues[i1] = vals[ind]
        eigenvectors[i1] = vecs[:, ind] 

    return eigenvalues, eigenvectors

def _spectrum_2d(
    hamiltonian_function: Callable[[float, float], np.ndarray],
    Brillouin_zone: Sequence[np.ndarray],
    num_bands: int
):
    #unpack BZ points
    K1s, K2s = Brillouin_zone[0], Brillouin_zone[1]
    N1, N2 = len(K1s), len(K2s)

    #initialize arrays to hold eigenvalues and eigenvectors
    eigenvalues = np.zeros((N1, N2, num_bands), dtype=np.float64)
    eigenvectors = np.zeros((N1, N2, num_bands, num_bands), dtype=np.complex128)

    for i1, k1 in enumerate(K1s):
        for i2, k2 in enumerate(K2s):
            vals, vecs = np.linalg.eigh(hamiltonian_function(k1, k2))

            ind = np.argsort(vals)  #sort eigenvalues and eigenvectors
            eigenvalues[i1, i2] = vals[ind]
            eigenvectors[i1, i2] = vecs[:, ind] 

    return eigenvalues, eigenvectors

def _spectrum_3d(
    hamiltonian_function: Callable[[float, float, float], np.ndarray],
    Brillouin_zone: Sequence[np.ndarray],
    num_bands: int
):
    #unpack BZ points
    K1s, K2s, K3s = Brillouin_zone[0], Brillouin_zone[1], Brillouin_zone[2]
    N1, N2, N3 = len(K1s), len(K2s), len(K3s)

    #initialize arrays to hold eigenvalues and eigenvectors
    eigenvalues = np.zeros((N1, N2, N3, num_bands), dtype=np.float64)
    eigenvectors = np.zeros((N1, N2, N3, num_bands, num_bands), dtype=np.complex128)

    for i1, k1 in enumerate(K1s):
        for i2, k2 in enumerate(K2s):
            for i3, k3 in enumerate(K3s):
                vals, vecs = np.linalg.eigh(hamiltonian_function(k1, k2, k3))

                ind = np.argsort(vals)  #sort eigenvalues and eigenvectors
                eigenvalues[i1, i2, i3] = vals[ind]
                eigenvectors[i1, i2, i3] = vecs[:, ind] 

    return eigenvalues, eigenvectors

