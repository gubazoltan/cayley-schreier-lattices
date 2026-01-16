import sympy as sp
from .linalg import Pauli
from sympy.physics.quantum import TensorProduct as TP


# helper functions for symmetry checks
def _rotate_c6(Hk, ksyms):
    """Rotate momenta of H(kx, ky) by 60 degrees (C6)."""
    k1, k2 = ksyms
    k1p, k2p = sp.symbols('k1p k2p', real=True)
    H_subs = Hk.subs({k1:k1p, k2:k2p})
    cosine, sine = sp.cos(sp.pi/3), sp.sin(sp.pi/3)
    H_subs = H_subs.subs({k1p:cosine*k1+sine*k2, k2p:-sine*k1+cosine*k2})
    return H_subs


def _shift_2pi(matrix:sp.Matrix, k: sp.Symbol)-> sp.Matrix:
    """Shift ``k`` by 2π and simplify."""
    return sp.simplify(matrix.subs({k:k+2*sp.pi}))

def check_C6(Hk: sp.Matrix, U: sp.Matrix, ksyms: list[sp.Symbol]) -> sp.Matrix:
    """Return H_C6(k) − H(k) for a C6 rotation."""
    # rotate kx, ky by 60 degrees 
    H_C6k = _rotate_c6(Hk, ksyms)

    # check invariance
    H_rot = U.H * H_C6k * U
    H_rot = H_rot.expand()

    diff = sp.simplify(H_rot - Hk)

    return diff


def check_periodicity(Hk, ksyms):
    """Print a warning if H(k) is not 2π-periodic in k1 or k2."""

    if not sp.simplify(_shift_2pi(Hk, ksyms[0]) - Hk) == sp.zeros(4):
        print("Hamiltonian is not periodic in k1 direction")

    if not sp.simplify(_shift_2pi(Hk, ksyms[1]) - Hk) == sp.zeros(4):
        print("Hamiltonian is not periodic in k2 direction")

def create_Gy_unitary(ksym): 
    #define the symmetry operator Gy
    s0, sx, sy, _ = Pauli()
    Aut = TP(s0, (sy-sx)/sp.sqrt(2))
    Uni = sp.Matrix([[sp.zeros(2,2), s0*sp.exp(sp.I*ksym)],[s0, sp.zeros(2,2)]])
    Gy = Aut @ Uni
    
    return Gy

def create_Tx_unitary(ksym):
    #define the symmetry operator Tx
    Z2 = sp.zeros(2,2)
    s0 = sp.eye(2)
    ek = sp.exp(sp.I*ksym)
    Tx = sp.BlockMatrix([[Z2, Z2, ek*s0, Z2],
                        [Z2, Z2, Z2, ek*s0],
                        [s0, Z2, Z2, Z2],
                        [Z2, s0, Z2, Z2]])
    
    return Tx