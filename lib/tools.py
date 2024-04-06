import numpy as np


def compute_fidelity(a, b) -> float:
    """
    Returns the fidelity of two quantum states, a and b.
    """
    return (a.dag() * b).norm() ** 2

def epsilon(energy : np.array, E_min:float, E_max:float):
    return (np.real(energy) - E_min)/(E_max - E_min)

def prepare_hook(H0, args):
    fidelity_, eps_ = [], []
    def hook_function(t, psi):
        # compute instantaneous GS
        H_now = H0(t=t, args=args)
        _, gs = H_now.groundstate()
        
        problem_energies = H_now.eigenenergies()
        E_min, E_max = min(problem_energies), max(problem_energies)
        
        fidelity_.append( compute_fidelity(gs, psi) )
        eps_.append( epsilon(psi.dag()*H_now*psi, E_min, E_max) )
    
    return fidelity_, eps_, hook_function