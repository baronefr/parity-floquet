
from lib.protocols.ua import *
import scipy


# general solver settings, if not overridden
opt = qutip.solver.Options(num_cpus=4)


# derivatives of A and B wrt lambda
derivs = (-1,1)

def beta_1(omega_0, alpha_1):
    """Beta_1 coefficient, as in ref paper."""
    return 2*omega_0*alpha_1

def beta_2(omega_0, alpha_1, alpha_2):
    """Beta_2 coefficient, as in ref paper."""
    return 2*omega_0*(24*alpha_2*omega_0*omega_0 + 3*alpha_1)




# %% [markdown]
# ### Floquet drivers

class claeys_l1:
    """Coefficients for Floquet driving (l=1) for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        
        osc = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        
        return AAA*osc + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        
        osc = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        
        return BBB*osc + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        coeff = np.sin(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        return AAA + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        coeff = np.sin(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        return BBB + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        eta_1 = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )

    def drive_cycle_error(H0, H0_partial, t :float, args :dict) -> float:
        """Error on a drive cycle for FE (l=1). Must be multiplied by the oscillation period!"""
        omega_0 = args['omega_0']
        alpha_1 = args['alpha_1']
        
        HH = H0(t, args=args)
        ddHH = H0_partial(t, args=args)
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        fact = np.abs( beta_1(omega_0, alpha_1=alpha_1(AAA,BBB)) )

        A = HH.norm()
        C = ddHH.norm() * (fact**2) * (args['scale_f_dot'](t, tau=args['tau']) **2)
        B = 2*np.sqrt( (HH*ddHH).norm() ) * fact * np.abs( args['scale_f_dot'](t, tau=args['tau']) )
        return 2*A + 4*B/np.pi + C
    
    def drive_amplitude(t,args):
        """Coefficient of dH in Floquet l=1."""
        omega_0 = args['omega_0']
        alpha_1 = args['alpha_1']
        AAA, BBB = A_f(t,args), B_f(t,args)
        coeff = 1*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        return coeff*args['scale_f_dot'](t, tau=args['tau'])





class claeys_l2:
    """Coefficients for Floquet driving (l=2) for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)

        osc = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t) * beta_1(omega_0, alpha_1=a_1) + \
            np.sin(3*omega*t) * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        
        return AAA*osc + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)

        osc = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t) * beta_1(omega_0, alpha_1=a_1) + \
            np.sin(3*omega*t) * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        
        return BBB*osc + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=2."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)
        coeff = np.sin(omega*t) * beta_1(omega_0, alpha_1=a_1) + \
            np.sin(3*omega*t) * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        return AAA + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=2."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)
        coeff = np.sin(omega*t) * beta_1(omega_0, alpha_1=a_1) + \
            np.sin(3*omega*t) * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        return BBB + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        
        eta_1 = (1 + (omega*np.cos(omega*t)/omega_0))
        coeff = np.sin(omega*t) * beta_1(omega_0, alpha_1=a_1) + \
            np.sin(3*omega*t) * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )

    def drive_cycle_error(H0, H0_partial, t :float, args :dict) -> float:
        """Error on a drive cycle for FE (l=2). Must be multiplied by the oscillation period!"""
        omega_0 = args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        HH = H0(t, args=args)
        ddHH = H0_partial(t, args=args)
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        
        a_1 = np.abs( alpha_1(AAA,BBB) )
        a_2 = np.abs( 3*alpha_1(AAA,BBB) + 24*omega_0*omega_0*alpha_2(AAA,BBB) )
        
        A = HH.norm()
        C = ddHH.norm() * (4*omega_0*omega_0) * (args['scale_f_dot'](t, tau=args['tau']) **2)
        B = 2*np.sqrt( (HH*ddHH).norm() ) * 2*omega_0 * np.abs( args['scale_f_dot'](t, tau=args['tau']) )
        return 2*A + 4*B*(a_1 + a_2)/np.pi + C*a_1*a_1 + C*a_2*a_2

    def drive_amplitude(t,args):
        """Test coefficient of dH in Floquet l=1."""
        omega_0 = args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)
        coeff = 1 * beta_1(omega_0, alpha_1=a_1) + \
            1 * beta_2(omega_0, alpha_1=a_1, alpha_2=a_2)
        return coeff*args['scale_f_dot'](t, tau=args['tau'])




# %%

def get_effective_FE(H_i, H_p, l):
    """Return the effective Floquet hamiltonian."""
    if l == 1:
        return qutip.QobjEvo([[H_i, claeys_l1.A_f_effective], [H_p, claeys_l1.B_f_effective]])
    elif l == 2:
        return qutip.QobjEvo([[H_i, claeys_l2.A_f_effective], [H_p, claeys_l2.B_f_effective]])
    else:
        raise ValueError(f'Floquet order l={l} not valid.')

def norm_integrator(HH, args, t0, dt, norm_type:str='fro', resolution:int=30):
    """Integrate the norm of HH in a time interval dt starting from time t0."""
    def integrand(tt):
        return HH(t0 + tt, args=args).norm(norm_type)
    integral, _ = scipy.integrate.quad(integrand,0, dt, limit=resolution)
    return integral

def error_effective_hamiltonian(HH, args, time):
    omega = args['omega']
    err = []
    for tt in time:
        err.append(  norm_integrator(HH, args, tt, dt = 2*np.pi/omega)  )
    return 2*np.array(err)



def norm_HFE(HH, time, args):
    """Computes 2||H_{FE}/2\pi||(t)"""
    raise DeprecationWarning('to remove')
    return np.array([ 2*((HH(tt, args=args)/(2*np.pi)).norm()) for tt in time ])


def check_norm_HFE(norms, omega, thres = 0.3):
    max_norm = np.max(norms)
    if max_norm/omega > thres:
        omega_suggestion = max_norm/thres
        print(f'suggested to increase omega to {omega_suggestion:.2e} (x{np.round(omega_suggestion/omega,2)})')
        return omega_suggestion
    else:
        print(f'omega is fine')
        return omega

def get_strobo_times(tau, omega, factor = 1):
    T = 2*np.pi/(factor*np.float128(omega))
    return np.arange(0, tau, T)

def get_number_full_oscillations(tau, omega) -> float:
    T = 2*np.pi/omega
    return tau/T





# %% [markdown]







# %% [markdown]
# ## run simulations




def run_Claeys_l1(H_i, H_p, time, Hargs, psi0 = None, schedule:str='sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, claeys_l1.A_f], [H_p, claeys_l1.B_f]])
    
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity_F1, epsilon_F1, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon_F1 = np.array(epsilon_F1).reshape(-1)
    
    return fidelity_F1, epsilon_F1


def run_Claeys_l2(H_i, H_p, time, Hargs, psi0 = None, schedule:str='sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, claeys_l2.A_f], [H_p, claeys_l2.B_f]])
    
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity_F2, epsilon_F2, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon_F2 = np.array(epsilon_F2).reshape(-1)
    
    return fidelity_F2, epsilon_F2


def return_Claeys_l1(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, claeys_l1.A_f], [H_p, claeys_l1.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs

def return_Claeys_l2(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, claeys_l2.A_f], [H_p, claeys_l2.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs


