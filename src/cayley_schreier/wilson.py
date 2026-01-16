import numpy as np
from typing import Tuple
from .bands import continuous_bands_1d


def wilson_loop_eigs(vecs: np.ndarray, axis: int = -3) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Wilson-loop phases and eigenvectors along a closed loop."""
    # move loop axis to -3: (..., L, n, n_occ)
    vecs = np.moveaxis(vecs, axis, -3)
    L = vecs.shape[-3]

    # Neighbor links along the loop
    Links = np.einsum("...ai,...aj->...ij", np.roll(vecs.conj(), -1, axis=-3), vecs)

    # Unitary projection via SVD
    U, _, Vh = np.linalg.svd(Links, full_matrices=False)
    F = np.einsum("...ik,...kj->...ij", U, Vh)

    # Wilson loop at the base point 
    R = np.empty_like(F)
    R[..., 0, :, :] = F[..., 0, :, :]
    for t in range(1, L):
        R[..., t, :, :] = F[..., t, :, :] @ R[..., t - 1, :, :]

    T = R[..., -1, :, :]  # total loop at base 0

    evals, V0 = np.linalg.eig(T)
    phases = np.angle(evals) / (2 * np.pi)
    wilson_basis = np.einsum("...mn, ...nk -> ...mk", vecs[..., 0, :, :], V0)

    wilson_basis, phases = continuous_bands_1d(wilson_basis, phases)

    return phases, wilson_basis