import numpy as np
from qutip import *


def Hamiltonian(omega_a, omega_m, g, n, a, sm, rwa):
    """
    Mendefinisikan Hamiltonian untuk model Jaynes-Cumming dengan parameter yang sesuai.

    Parameter
    ---------
    omega_a : :int/float
            Merupakan ungkapan frekuensi transisi atom

    omega_m : :int/float
            Merupakan ungkapan frekuensi meden

    g : :int/float
        Merupakan ungkapan parameter kuat interaksi sistem atom-meda

    n : :int
        Merupakan ungkapan parameter jumlah N medan
        
    a : :Qobj
        Operator anihilasi

    sm : :Qobj
        Operator sigma_negatif

    rwa : :bool
        Merupakan parameter RWA.
        True : menggunakan RWA
        False : tidak menggunakan RWA

    """
    # Hamiltonian
    if rwa:
        H = omega_m * a.dag() * a + 0.5 * omega_a * commutator(sm.dag(), sm) + g * (a.dag() * sm + a * sm.dag())
    else:
        H = omega_m * a.dag() * a + 0.5 * omega_a * commutator(sm.dag(), sm) + g * (a.dag() + a) * (sm + sm.dag())
    return H
