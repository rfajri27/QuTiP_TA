import numpy as np
from qutip import *


def op_collapse(gamma, kappa, a, sm, n_th=0.0):
    """
    Operator kerutuhan (collapse) akan digunakan sebagai parameter pada
    persamaan master.

    Parameter
    ---------
    gamma : : float
        Merupakan ungkapan laju disipasi pada atom

    kappa : : float
        Merupakan ungkapan laju disipasi pada medan

    a : :Qobj
        Operator anihilasi

    sm : :Qobj
        Operator sigma_negatif

    n_th : 0.0 : float
        Merupakan ungkapan jumlah rata-rata eksitasi thermal bath

    Return
    ---------
    Output : :Qobj
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
