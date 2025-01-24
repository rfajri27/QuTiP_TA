import numpy as np
from qutip import *


def Hamiltonian(omega_a, omega_m, g, n, a, sm, rwa):
    """
    Mendefinisikan Hamiltonian untuk model Jaynes-Cumming dengan parameter yang sesuai.

    Parameter
    ---------
    omega_a ::int/float
            Merupakan ungkapan frekuensi transisi atom

    omega_m ::int/float
            Merupakan ungkapan frekuensi meden

    g ::int/float
        Merupakan ungkapan parameter kuat interaksi sistem atom-meda

    n ::int
        Merupakan ungkapan parameter jumlah N medan

    a ::Qobj
        Operator anihilasi

    sm ::Qobj
        Operator sigma_negatif

    rwa ::bool
        Merupakan parameter RWA.
        True : menggunakan RWA
        False : tidak menggunakan RWA

    Return
    ---------
    Output : :Qobj
        Keluaran berupa ungkapan Hamiltonian untuk model Jaynes-Cumming
    """

    # Hamiltonian
    if rwa:
        H = omega_m * a.dag() * a + 0.5 * omega_a * commutator(sm.dag(), sm) + \
            g * (sm.dag() * a + sm * a.dag())
    else:
        H = omega_m * a.dag() * a + 0.5 * omega_a * commutator(sm.dag(), sm) + \
            g * (sm.dag() + sm) * (a.dag() + a)
    return H


def op_collapse(gamma, kappa, a, sm, n_th=0.0):
    """
    Operator kerutuhan (collapse) akan digunakan sebagai parameter pada
    persamaan master.

    Parameter
    ---------
    gamma :: float
        Merupakan ungkapan laju disipasi pada atom

    kappa :: float
        Merupakan ungkapan laju disipasi pada medan

    a :: Qobj
        Operator anihilasi

    sm :: Qobj
        Operator sigma_negatif

    n_th : 0.0 : float
        Merupakan ungkapan jumlah rata-rata eksitasi thermal bath

    Return
    ---------
    Output :: Qobj
        Keluaran berupa ungkapan operator kerutuhan (collapse)
    """

    c_ops = []

    # relaksasi medan
    rate = kappa * (1 + n_th)
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a)

    # operator kerutuhan medan jika T > 0
    rate = kappa * n_th
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a.dag())

    # relaksasi atom
    rate = gamma
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * sm)

    return c_ops


def Wigner(psi, xvec, yvec):
    """ 
    Menghitung fungsi Wigner untuk suatu vektor keadaan pada xvec + i*yvec

    Parameter
    ---------
    psi :: Qobj
        Merupakan ungkapan vektor keadaan

    xvec :: array
        Koordinat x

    yvec :: array
        Koordinat y

    Return
    ---------
    Output :: array
        Keluaran berupa ungkapan fungsi Wigner

    """

    rho = ptrace(psi, 0)

    W = wigner(rho, xvec, yvec)

    return W


def Entropy(psi):
    """
    Menghitung entropi Von-Neumann dari suatu densitas matriks

    Parameter
    ---------
    psi :: Qobj
         Merupakan ungkapan vektor keadaan

    Return
    ---------
    Output :: float
        Keluaran berupa ungkapan entropi dari suatu densitas matriks
    """

    rho = ptrace(psi, 1)

    S = entropy_vn(rho, 2)

    return S
