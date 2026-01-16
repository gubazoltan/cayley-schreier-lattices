"""Public interface for the cayley_schreier package."""
from sympy.physics.quantum import TensorProduct as TP

from .spectrum import spectrum
from .hamiltonians import create_triangular_hamiltonian, create_honeycomb_hamiltonian
from .linalg import Pauli, quaternion_e_irrep
from .bands import continuous_bands_1d
from .topology import compute_partial_polarization
from .surface import get_surface_hamiltonian
from .symmetry import check_C6, check_periodicity
from .wilson import wilson_loop_eigs

__all__ = [
    "spectrum",
    "create_triangular_hamiltonian",
    "create_honeycomb_hamiltonian",
    "Pauli",
    "quaternion_e_irrep",
    "continuous_bands_1d",
    "compute_partial_polarization",
    "get_surface_hamiltonian",
    "check_C6",
    "check_periodicity",
    "wilson_loop_eigs",
]
