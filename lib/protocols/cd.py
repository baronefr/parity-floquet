
from lib.protocols.ua import *
import lib.numcd as numcd


# %% [markdown]
# ## CD protocol

def agp1_f(t, args):
    alpha_1 = args['alpha_1']
    AAA, BBB = A_f(t,args), B_f(t,args)
    a_1 = alpha_1(AAA, BBB)
    return 1j*args['scale_f_dot'](t, tau=args['tau'])*a_1

def agp2a_f(t, args):
    alpha_2 = args['alpha_2']
    AAA, BBB = A_f(t,args), B_f(t,args)
    a_2 = alpha_2(AAA, BBB)
    return 1j*args['scale_f_dot'](t, tau=args['tau'])*a_2*AAA*AAA

def agp2b_f(t, args):
    alpha_2 = args['alpha_2']
    AAA, BBB = A_f(t,args), B_f(t,args)
    a_2 = alpha_2(AAA,BBB)
    return 1j*args['scale_f_dot'](t, tau=args['tau'])*a_2*AAA*BBB

# note: no need to define agp2c_f, it is equal to agp2b_f

def agp2d_f(t, args):
    alpha_2 = args['alpha_2']
    AAA, BBB = A_f(t,args), B_f(t,args)
    a_2 = alpha_2(AAA, BBB)
    return 1j*args['scale_f_dot'](t, tau=args['tau'])*a_2*BBB*BBB



def run_CD_l1(H_i, H_p, time, Hargs, psi0 = None, schedule:str='sin', options = opt):
    """Run CD annealing schedule at order l=1."""
    H0 = make_H0(H_i, H_p)
    AGP_l1, _ = numcd.CD_l2_agp(H_i, H_p)
    H_CD1 = H0 + qutip.QobjEvo([[AGP_l1, agp1_f]])

    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity_C1, epsilon_C1, hf = tools.prepare_hook(H0, Hargs) 
    res = qutip.sesolve(H_CD1, psi0, time, e_ops=hf, args=Hargs, options=options)
    epsilon_C1 = np.array(epsilon_C1).reshape(-1)
    
    return fidelity_C1, epsilon_C1


def run_CD_l2(H_i, H_p, time, Hargs, psi0 = None, schedule:str='sin', options = opt):
    """Run CD annealing schedule at order l=2."""
    H0 = make_H0(H_i, H_p)
    AGP_l1, AGP_l2_list = numcd.CD_l2_agp(H_i, H_p)
    H_CD2 = H0 + qutip.QobjEvo([ 
        [ AGP_l1, agp1_f], 
        [ AGP_l2_list[0], agp2a_f], [ AGP_l2_list[1], agp2b_f], 
        [ AGP_l2_list[2], agp2b_f], [ AGP_l2_list[3], agp2d_f]]
    )
    
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity_C2, epsilon_C2, hf = tools.prepare_hook(H0, Hargs) 
    res = qutip.sesolve(H_CD2, psi0, time, e_ops=hf, args=Hargs, options=options)
    epsilon_C2 = np.array(epsilon_C2).reshape(-1)
    
    return fidelity_C2, epsilon_C2


def return_CD_l1(H_i, H_p, Hargs, schedule:str='sin'):
    H0 = make_H0(H_i, H_p)
    AGP_l1, _ = numcd.CD_l2_agp(H_i, H_p)
    H_CD1 = H0 + qutip.QobjEvo([[AGP_l1, agp1_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_CD1, Hargs

def return_CD_l2(H_i, H_p, Hargs, schedule:str='sin'):
    H0 = make_H0(H_i, H_p)
    AGP_l1, AGP_l2_list = numcd.CD_l2_agp(H_i, H_p)
    H_CD2 = H0 + qutip.QobjEvo([ 
        [ AGP_l1, agp1_f], 
        [ AGP_l2_list[0], agp2a_f], [ AGP_l2_list[1], agp2b_f], 
        [ AGP_l2_list[2], agp2b_f], [ AGP_l2_list[3], agp2d_f]]
    )
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_CD2, Hargs


