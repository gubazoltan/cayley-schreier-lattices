import numpy as np

from scipy.optimize import linear_sum_assignment as lsa


def continuous_bands_1d(
    vecs: np.ndarray, vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorders the vectors and values to ensure continuity of bands.
    """
    vecs_out = vecs.copy()
    vals_out = vals.copy()

    for i in range(len(vals) - 1):
        v_prev = vecs_out[i]
        v_next = vecs_out[i + 1]
        vals_next = vals_out[i + 1]

        # Compute overlap matrix and optimal assignment
        overlap = np.abs(v_prev.conj().T @ v_next)
        _, assignment = lsa(-overlap)

        # Reorder next vectors and values
        vecs_out[i + 1] = v_next[:, assignment]
        vals_out[i + 1] = vals_next[assignment]

    return vecs_out, vals_out