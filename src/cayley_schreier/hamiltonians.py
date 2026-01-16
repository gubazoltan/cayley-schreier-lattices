import sympy as sp
from .linalg import Pauli, quaternion_e_irrep
from sympy.physics.quantum import TensorProduct as TP

# Helpers inside module
def _phase(k):
    I = sp.I
    return sp.exp(I*k), sp.exp(-I*k)

def _nn_vectors_12():
    b1 = sp.Matrix([sp.Rational(2,3), sp.Rational(1,3)]) 
    b2 = sp.Matrix([-sp.Rational(1,3), sp.Rational(1,3)]) 
    b3 = sp.Matrix([-sp.Rational(1,3), -sp.Rational(2,3)]) 
    return b1, b2, b3

def _nn_vectors_xy():
    b1 = sp.Matrix([sp.sqrt(3)*sp.Rational(1,2), -sp.Rational(1,2)])
    b2 = sp.Matrix([0,1])
    b3 = sp.Matrix([-sp.sqrt(3)*sp.Rational(1,2), -sp.Rational(1,2)])
    return b1, b2, b3

def _exponential_rewrite(matrix: sp.Matrix): 
    """Rewrite all entries of a SymPy matrix in terms of exponentials."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = sp.simplify(matrix[i, j].rewrite(sp.exp)).expand()
    return matrix

"""Hamiltonian builders for a triangular ladder model.

k  : crystal momentum
t1 : primary hopping amplitude
t2 : secondary (diagonal/fold) hopping amplitude

Functions return Hermitian SymPy matrices. Variants 'a', 'b', and 'folded'.
"""

def create_triangular_hamiltonian(symbols, model = "a"):
    """Factory selecting one of the ladder Hamiltonian variants."""
    if model == "a":
        return _ladder_a(symbols)
    elif model == "b":
        return _ladder_b(symbols)
    elif model == "folded":
        return _ladder_folded(symbols)
    else:
        raise ValueError("model must be either 'a', 'b', or 'folded'")

def _ladder_a(symbols):
    """Variant A: Corresponds to model a in the article."""
    k, t1, t2 = symbols
    mats = quaternion_e_irrep()

    # construct the Hamiltonian
    hAA = t1 * sp.exp(sp.I * k) * mats[1] + t1 * sp.exp(-sp.I * k) * mats[1].H
    hBA = t1 * mats[0] + t2 * sp.exp(-sp.I * k) * mats[0]
    hAB = hBA.H
    hBB = t1 * sp.exp(sp.I * k) * mats[2].H + t1 * sp.exp(-sp.I * k) * mats[2]

    Hk = sp.Matrix([[hAA, hAB], [hBA, hBB]])

    assert Hk.is_hermitian, "Hamiltonian is not Hermitian!"

    return Hk


def _ladder_b(symbols):
    """Variant B: Corresponds to model b in the article."""
    k, t1, t2 = symbols
    mats = quaternion_e_irrep()

    # construct the Hamiltonian
    hAA = t1 * sp.exp(sp.I * k) * mats[3] + t1 * sp.exp(-sp.I * k) * mats[3].H
    hBA = t1 * mats[3] + t2 * sp.exp(-sp.I * k) * mats[1]
    hAB = hBA.H
    hBB = t1 * sp.exp(sp.I * k) * mats[0].H + t1 * sp.exp(-sp.I * k) * mats[0]

    Hk = sp.Matrix([[hAA, hAB], [hBA, hBB]]) 

    assert Hk.is_hermitian, "Hamiltonian is not Hermitian!"

    return Hk

def _ladder_folded(symbols):
    """Folded variant: corresponds to the folded model in the article."""
    k, t1, t2 = symbols
    ek, emk = _phase(k)
    tau = t2 / t1
    s0, sx, _, sz = Pauli()
    z2 = sp.zeros(2)
    I = sp.I
    blocks = [
        [z2,          I*sz,                I*(1 - ek)*sz,       I*tau*ek*sx],
        [-I*sz,       z2,                  -I*tau*sx,           (1 + ek)*s0],
        [I*(emk-1)*sz, I*tau*sx,           z2,                  I*sz],
        [-I*tau*emk*sx, (1 + emk)*s0,      -I*sz,               z2],
    ]
    Hk = t1 * sp.BlockMatrix(blocks).as_explicit()
    assert Hk.is_hermitian
    return Hk

"""Hamiltonian builders for the honeycomb model.

k1, k2 : crystal momenta
t1, t2 : primary and secondary hopping amplitudes
m : sublattice potential 

Function return Hermitian SymPy matrices.
"""

def create_honeycomb_hamiltonian(symbols, basis = "12"):
    """Factory selecting the honeycomb Hamiltonian."""
    k1, k2, t1, t2, m = symbols
    s0, sx, sy, sz = Pauli()
    #express the nearest neighbor vectors in the chosen basis
    if basis == "12": 
        b1, b2, b3 = _nn_vectors_12()
    elif basis == "xy": 
        b1, b2, b3 = _nn_vectors_xy()
    else: 
        raise ValueError("basis must be either '12' or 'xy'")
    
    #next-nearest neighbor vectors
    a1 = b1 - b2
    a2 = b2 - b3
    a3 = b3 - b1

    kvec = sp.Matrix([k1, k2])

    #define the terms in the Hamiltonian
    H_mass = m * TP(sz, s0) 
    H_SOC = -2* t2* (
            sp.sin(kvec.dot(a1)) * TP(sz, sx)
            + sp.sin(kvec.dot(a2)) * TP(sz, sy)
            + sp.sin(kvec.dot(a3)) * TP(sz, sz)
        )
    H_nn = t1 * (
        TP(sx, s0)*(sp.cos(kvec.dot(b1)) + sp.cos(kvec.dot(b2)) + sp.cos(kvec.dot(b3)))
        + TP(sy, s0)*(sp.sin(kvec.dot(b1)) + sp.sin(kvec.dot(b2)) + sp.sin(kvec.dot(b3)))
    )

    #construct the full Hamiltonian
    Hk = H_mass + H_SOC + H_nn
    
    #optimize the expression with exponentials
    Hk = _exponential_rewrite(Hk)

    #assure Hermiticity
    assert Hk.H == Hk, "Hamiltonian is not Hermitian!"
    
    return Hk