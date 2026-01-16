import sympy as sp


def Pauli():
    """Return the Pauli matrices as [s0, sx, sy, sz]."""
    s0 = sp.Matrix([[1, 0], [0, 1]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    return [s0, sx, sy, sz]


def quaternion_e_irrep():
    """Return the 2D irrep of the quaternion group as a list of matrices."""
    s0, sx, sy, sz = Pauli()
    return [s0, -sp.I * sx, -sp.I * sy, -sp.I * sz, -s0, sp.I * sx, sp.I * sy, sp.I * sz]