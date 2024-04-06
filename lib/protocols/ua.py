
import qutip
import numpy as np

import lib.protocols as proto
import lib.tools as tools


# general solver settings, if not overridden
opt = qutip.solver.Options(num_cpus=4)




# %% [markdown]
# ## UA protocol

def A_f(t, args):
    """Schedule function: H_i coefficient."""
    return 1 - args['scale_f'](t, tau=args['tau'])

def B_f(t, args):
    """Schedule function: H_p coefficient."""
    return args['scale_f'](t, tau=args['tau'])

def make_H0(H_i, H_p) -> qutip.QobjEvo:
    """Make hamiltonian, interpolating H_i and H_p via A_f, B_f."""
    return qutip.QobjEvo([[H_i, A_f], [H_p, B_f]])


def run_UA(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    """Run unoptimized annealing schedule."""
    
    H0 = make_H0(H_i, H_p)
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    # compute init state, if not provided
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    
    fidelity_UA, epsilon_UA, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H0, psi0, time, e_ops=hf, args=Hargs, options=options)
    epsilon_UA = np.array(epsilon_UA).reshape(-1)

    return fidelity_UA, epsilon_UA


def return_UA(H_i, H_p, Hargs, schedule:str='sin'):
    H0 = make_H0(H_i, H_p)
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H0, Hargs


