
# %%

import numpy as np
import qutip



# %% LHZ

# generic term builder
def build_LHZ_hamiltonian(J:np.array, C:np.array, interactions:np.array):
    """Build Hamiltonian for the LHZ model."""
    
    H_x, H_z, H_c = 0, 0, 0
    N = len(J)
    
    # build H_x with single site sigma_x
    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = qutip.sigmax()
        H_x += qutip.tensor(site)  # NOTE: this should be - to get a |+> GS

    # build H_p with single site sigma_z
    for ii in range(N):
        site = [ qutip.qeye(2) for _ in range(N) ]
        site[ii] = J[ii]*qutip.sigmaz()
        H_z += qutip.tensor(site)

    # build H_p interactions
    for ii, tt in enumerate(interactions):
        site = [ (qutip.sigmaz()) if jj in tt else qutip.qeye(2) for jj in range(N) ]
        H_c += C[ii]*qutip.tensor(site)
    
    return H_x, H_z, H_c



def get_LHZ_06(coeff:str):
    N = 6
    interactions = [ [1,2,3,4], [0,1,3], [3,4,5] ]
    C_neighbor_interactions = [ [1,2], [0,2], [0,1] ] # list of neighbors for each vertex

    if coeff == "rand":
        Js = np.random.randint(0,2,size=N)
        Cs = np.random.randint(0,4,size=len(interactions))
        
    elif coeff == "homo":
        Js = np.ones(N)*1
        Cs = np.ones(len(interactions))*-2
    
    elif coeff == "c1":
        Js = np.array([-1,-1,1,-1,-1,1])
        Cs = -2*np.ones(3)
    
    else:
        raise ValueError(f"unknown coefficient configuration ({coeff})")
    
    return Js, Cs




def get_LHZ_10(coeff:str):
    N = 10
    interactions = [ [1,2,4,5], [2,3,5,6], [5,6,7,8], [0,1,4], [4,5,7], [7,8,9] ]
    C_neighbor_interactions = [ [1,2,3,4], [0,2,4], [0,1,4,5], [0,4], [0,1,2,3,5], [2,4] ] # list of neighbors for each vertex 
    
    if coeff == "rand":
        Js = np.random.randint(0,2,size=N)
        Cs = np.random.randint(0,4,size=len(interactions))
        
    elif coeff == "homo":
        Js = np.ones(N)*1
        Cs = np.ones(len(interactions))*-2
    
    else:
        raise ValueError(f"unknown coefficient configuration ({coeff})")
    
    return Js, Cs



def check_LHZ_energy(H_z, H_p):
    _, hzgs = H_z.groundstate()
    Egs = hzgs.dag()*H_p*hzgs

    energies = H_p.eigenenergies()
    return np.real( ((Egs - min(energies))/(max(energies)-min(energies))).data[0,0] )



class lcd:

    # first order
    def ansatz_Y(N:int):
        """Make the LCD ansatz \sum_i \alpha_i \sigma_y^i."""
        sigY_ansatz = []
        for ii in range(N):
            site = [ qutip.qeye(2) for _ in range(N) ]
            site[ii] = qutip.sigmay()
            sigY_ansatz.append( qutip.tensor(site) )
        return sigY_ansatz

    def ansatz_homoY(N:int):
        """Make the LCD ansatz \alpha \sum_i \sigma_y^i."""
        sigY_ansatz_single = 0
        for ii in range(N):
            site = [ qutip.qeye(2) for _ in range(N) ]
            site[ii] = qutip.sigmay()
            sigY_ansatz_single += qutip.tensor(site)
        return [ sigY_ansatz_single ]


    # second order
    def ansatz_YXXX(N:int, interactions):
        ans = []
        for inter in interactions:
            constr_op = 0
            for jj in inter:
                site = [ (qutip.sigmax()) if ii in inter else qutip.qeye(2) for ii in range(N) ]
                site[jj] = qutip.sigmay()
                constr_op += qutip.tensor(site)
            ans.append(constr_op)
        return ans

    def ansatz_YZZZ(N:int, interactions):
        ans = []
        for inter in interactions:
            constr_op = 0
            for jj in inter:
                site = [ (qutip.sigmaz()) if ii in inter else qutip.qeye(2) for ii in range(N) ]
                site[jj] = qutip.sigmay()
                constr_op += qutip.tensor(site)
            ans.append(constr_op)
        return ans
    
    def ansatz_XYYY(N:int, interactions):
        ans = []
        for inter in interactions:
            constr_op = 0
            for jj in inter:
                site = [ (qutip.sigmay()) if ii in inter else qutip.qeye(2) for ii in range(N) ]
                site[jj] = qutip.sigmax()
                constr_op += qutip.tensor(site)
            ans.append(constr_op)
        return ans
    
    def ansatz_ZYYY(N:int, interactions):
        ans = []
        for inter in interactions:
            constr_op = 0
            for jj in inter:
                site = [ (qutip.sigmay()) if ii in inter else qutip.qeye(2) for ii in range(N) ]
                site[jj] = qutip.sigmaz()
                constr_op += qutip.tensor(site)
            ans.append(constr_op)
        return ans




