import qutip
import numpy as np
import lib.lhz as lhz


def model_lhz6_medium():
    """LHZ (N=6) with assigned coefficients."""
    N = 6
    interactions = [ [1,2,3,4], [0,1,3], [3,4,5] ]

    Js = np.array([-1,-1,1,-1,-1,1])  # eps = 0.3
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p

def model_lhz6_easy():
    """LHZ (N=6) with assigned coefficients."""
    N = 6
    interactions = [ [1,2,3,4], [0,1,3], [3,4,5] ]

    Js = np.array([ 1,-1,-1, 1, 1,-1])  # eps = 0.0  (line 40)
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p

def model_lhz6_hard():
    """LHZ (N=6) with assigned coefficients."""
    N = 6
    interactions = [ [1,2,3,4], [0,1,3], [3,4,5] ]

    Js = np.array([ 1, 1, 1, 1,-1,-1])  # eps = 0.555 (line 62)
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p


def model_lhz6_homo():
    """LHZ (N=6) with homogeneous coefficients."""
    N = 6
    interactions = [ [1,2,3,4], [0,1,3], [3,4,5] ]

    Js = np.ones(N)
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p


def model_lhz10_medium():
    """LHZ (N=10) with assigned coefficients."""
    N = 10
    interactions = [  [0,1,4], [4,5,7], [7,8,9], [1,2,4,5], [2,3,5,6], [5,6,7,8] ]

    Js = np.array([-1,-1, 1,-1,-1, 1,-1,-1,-1, 1])  # eps = 0.27 (line 147)
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p

def model_lhz10_easy():
    """LHZ (N=10) with assigned coefficients."""
    N = 10
    interactions = [  [0,1,4], [4,5,7], [7,8,9], [1,2,4,5], [2,3,5,6], [5,6,7,8] ]

    Js = np.array([ 1,-1, 1,-1, 1,-1, 1, 1,-1, 1])  # eps = 0.0 (line 687)
    Cs = -2*np.ones( len(interactions) )

    H_i, H_z, H_c = lhz.build_LHZ_hamiltonian(C=Cs, J=Js, interactions=interactions)
    H_p = H_z + H_c
    return H_i, H_p




def model_ising(N = 6, x=-1, h=-1, J=-2):
    """Ising model annealing"""
    H_i, H_z, H_int = 0, 0, 0

    # initial Hamiltonian
    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmax()
        H_i += x*qutip.tensor(site)

    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmaz()
        H_z += h*qutip.tensor(site)

    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmaz()
        site[(ii + 1) % N] = qutip.sigmaz()
        H_int += J*qutip.tensor(site)

    H_p = H_z + H_int
    return H_i, H_p




def model_3level(J:float=1, h:float=2):
    """3 level system annealing, inspired by ref paper."""
    H_i = 0
    N :int = 2
    
    # initial Hamiltonian
    H_i += -2*J*qutip.tensor([qutip.sigmaz(),qutip.sigmaz()])
    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmaz()
        H_i += -h*qutip.tensor(site)
    
    # problem Hamiltonian
    H_p = H_i  # the other members are constant
    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmax()
        H_p += 2*h*qutip.tensor(site)
    
    return H_i, H_p