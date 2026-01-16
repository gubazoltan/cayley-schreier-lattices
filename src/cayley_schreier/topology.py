import numpy as np


def compute_partial_polarization(vecs):
    """Compute the partial polarization from a loop of Bloch states."""
    assert vecs.ndim == 2, "Input vecs must be a 2D array (N, dim)."

    # shifted vectors
    vecs_shifted = np.roll(vecs, -1, axis=0)
    links = np.einsum('ij,ij->i', np.conj(vecs_shifted), vecs)
    links /= np.abs(links)

    wilson_loop = np.prod(links)
    return np.angle(wilson_loop) / (2 * np.pi)