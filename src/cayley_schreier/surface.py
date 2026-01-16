"""
This module builds on the fact the Hamiltonians are constructed 
from sympy matrices and are written in terms of exponentials. 
This allows us to extract the hopping matrices by integrating
over the Brillouin zone.
"""

import sympy as sp


def _block(i, num_bands):
    """Slice for the i-th block of size ``num_bands``."""
    return slice(i * num_bands, (i + 1) * num_bands)


def get_surface_hamiltonian(
    hamiltonian: sp.Matrix, 
    k: sp.Symbol, 
    system_size: int, 
    num_bands: int
) -> sp.Matrix:
    """
    Get the surface Hamiltonian for open boundary conditions.
    Only for nearest neighbor hopping. Works for any dimensional lattice,
    since the other momenta are symbols in the Hamiltonian.
    """

    # integrate out the k dependence to get the hopping matrices
    T_pos = sp.integrate(hamiltonian * sp.exp(-sp.I * k), (k, -sp.pi, sp.pi)) / (
        2 * sp.pi
    )
    T_neg = sp.integrate(hamiltonian * sp.exp(+sp.I * k), (k, -sp.pi, sp.pi)) / (
        2 * sp.pi
    )

    # simplify the hopping matrices
    T_pos = T_pos.rewrite(sp.exp).simplify()
    T_neg = T_neg.rewrite(sp.exp).simplify()

    # on-site term in real space
    H_diag = hamiltonian - (T_pos * sp.exp(sp.I * k) + T_neg * sp.exp(-sp.I * k))
    H_diag = H_diag.rewrite(sp.exp).simplify().expand()
    H_diag = sp.nsimplify(H_diag, tolerance=1e-8)  # get rid of small numerical errors

    assert k not in H_diag.free_symbols, "H_diag should not depend on k"

    # assemble real-space Hamiltonian with open boundaries
    H_realspace = sp.zeros(system_size * num_bands, system_size * num_bands)
    for i in range(system_size):
        H_realspace[_block(i, num_bands), _block(i, num_bands)] = H_diag[:, :]
        if i > 0:
            H_realspace[_block(i - 1, num_bands), _block(i, num_bands)] = T_neg[:, :]
        if i < system_size - 1:
            H_realspace[_block(i + 1, num_bands), _block(i, num_bands)] = T_pos[:, :]
    return H_realspace
