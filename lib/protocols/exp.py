
from lib.protocols.ua import *

# derivatives of A and B wrt lambda
derivs = (-1,1)

def beta_1(omega_0, alpha_1):
    """Beta_1 coefficient, as in ref paper."""
    return 2*omega_0*alpha_1




# %%

class ord2_l1:
    """Coefficients for Floquet driving (l=1) at second order, with cosines, for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        
        return AAA*osc + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        
        return BBB*osc + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        coeff = np.cos(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        return AAA + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        coeff = np.cos(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
        return BBB + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*beta_1(omega_0, alpha_1=alpha_1(AAA,BBB))
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

    
def run_ord2_l1(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord2_l1.A_f], [H_p, ord2_l1.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
        
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord2_l1(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord2_l1.A_f], [H_p, ord2_l1.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs





# %%

cutoff_div = 10

class ord3_l1:
    """Coefficients for Floquet driving (l=1) at third order, for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*2*omega_0*a_1 -\
            6*a_1*omega_0*np.cos(3*omega*t) +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA*osc + derivs[0]*coeff*lam_dot

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*2*omega_0*a_1 -\
            6*a_1*omega_0*np.cos(3*omega*t) +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB*osc + derivs[1]*coeff*lam_dot


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 -\
            6*a_1*omega_0*np.cos(3*omega*t) +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA + derivs[0]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 -\
            6*a_1*omega_0*np.cos(3*omega*t) +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 -\
            6*a_1*omega_0*np.cos(3*omega*t) +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )


def run_ord3_l1(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord3_l1.A_f], [H_p, ord3_l1.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord3_l1(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord3_l1.A_f], [H_p, ord3_l1.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs









# %%

class ord3v2_l1:
    """Coefficients for Floquet driving (l=1) at third order, for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*2*omega_0*a_1 +\
            ( np.cos(omega*t)  )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA*osc + derivs[0]*coeff*lam_dot

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = np.cos(omega*t)*2*omega_0*a_1 +\
            ( np.cos(omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB*osc + derivs[1]*coeff*lam_dot


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 +\
            ( np.cos(omega*t)  )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA + derivs[0]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 +\
            ( np.cos(omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1 = args['alpha_1']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = np.cos(omega*t)*2*omega_0*a_1 +\
            ( np.cos(omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )


def run_ord3v2_l1(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord3v2_l1.A_f], [H_p, ord3v2_l1.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord3v2_l1(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord3v2_l1.A_f], [H_p, ord3v2_l1.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs




# %%

class ord2_l2:
    """Coefficients for Floquet driving (l=2) for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)

        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0*a_1*( np.cos(omega*t) - 3* np.cos(3*omega*t) ) - \
            48 * np.cos(3*omega*t) * a_2 * (omega_0**3)
        
        return AAA*osc + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)

        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0*a_1*( np.cos(omega*t) - 3* np.cos(3*omega*t) ) - \
            48 * np.cos(3*omega*t) * a_2 * (omega_0**3)
        
        return BBB*osc + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=2."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)
        coeff = 2*omega_0*a_1*( np.cos(omega*t) - 3* np.cos(3*omega*t) ) - \
            48 * np.cos(3*omega*t) * a_2 * (omega_0**3)
        return AAA + derivs[0]*coeff*args['scale_f_dot'](t, tau=args['tau'])

    def B_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=2."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA, BBB)
        a_2 = alpha_2(AAA, BBB)
        coeff = 2*omega_0*a_1*( np.cos(omega*t) - 3* np.cos(3*omega*t) ) - \
            48 * np.cos(3*omega*t) * a_2 * (omega_0**3)
        return BBB + derivs[1]*coeff*args['scale_f_dot'](t, tau=args['tau'])


    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=2)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0*a_1*( np.cos(omega*t) - 3* np.cos(3*omega*t) ) - \
            48 * np.cos(3*omega*t) * a_2 * (omega_0**3)
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )


def run_ord2_l2(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord2_l2.A_f], [H_p, ord2_l2.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord2_l2(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord2_l2.A_f], [H_p, ord2_l2.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs



# %%

class ord3_l2:
    """Coefficients for Floquet driving (l=2) at third order, for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t)-5*np.cos(5*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) + 5*np.cos(5*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA*osc + derivs[0]*coeff*lam_dot

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t)-5*np.cos(5*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) + 5*np.cos(5*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB*osc + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t)-5*np.cos(5*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) + 5*np.cos(5*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA + derivs[0]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t)-5*np.cos(5*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) + 5*np.cos(5*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t)-5*np.cos(5*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) + 5*np.cos(5*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )


def run_ord3_l2(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord3_l2.A_f], [H_p, ord3_l2.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord3_l2(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord3_l2.A_f], [H_p, ord3_l2.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs




# %%

class ord3v2_l2:
    """Coefficients for Floquet driving (l=2) at third order, for H(l) = (1-l)H_i + l H_p."""
    
    def A_f(t, args:dict):
        """Coefficient of H_i for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA*osc + derivs[0]*coeff*lam_dot

    def B_f(t, args:dict):
        """Coefficient of H_p for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        osc = (1 - (omega*np.sin(omega*t)/omega_0))
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB*osc + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])


    def A_f_effective(t, args:dict):
        """Coefficient of H_i in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return AAA + derivs[0]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    def B_f_effective(t, args:dict):
        """Coefficient of H_p in effective Floquet l=1."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        term = ( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        return BBB + derivs[1]*np.nan_to_num(coeff)*args['scale_f_dot'](t, tau=args['tau'])
    
    
    def effective_time_integrand(t, args):
        """Effective time integrand for Floquet driving (l=1)."""
        omega, omega_0 = args['omega'], args['omega_0']
        alpha_1, alpha_2 = args['alpha_1'], args['alpha_2']
        
        AAA, BBB = A_f(t,args), B_f(t,args)
        a_1 = alpha_1(AAA,BBB)
        a_2 = alpha_2(AAA,BBB)
        lam_dot = args['scale_f_dot'](t, tau=args['tau'])
        lam_dotdot = args['scale_f_dotdot'](t, tau=args['tau'])
        
        eta_1 = (1 - (omega*np.sin(omega*t)/omega_0))
        
        term = np.array( 2 + (2*np.pi*lam_dotdot/(lam_dot*omega)) )
        term[ (1/term) > cutoff_div ] = 1/cutoff_div
        term[ (1/term) <-cutoff_div ] = -1/cutoff_div
        
        coeff = 2*omega_0* (np.cos(omega*t)-3*np.cos(3*omega*t)) *a_1 -\
            48*(omega_0**3)*( np.cos(3*omega*t) )*a_2 +\
            ( np.cos(omega*t) -3*np.cos(3*omega*t) )/(omega_0*term)
        coeff = np.nan_to_num(coeff)
        
        eta_2 = coeff * args['scale_f_dot'](t, tau=args['tau'])
        
        return np.abs( eta_1*AAA - eta_2 ) + np.abs( eta_1*BBB + eta_2 )


def run_ord3v2_l2(H_i, H_p, time, Hargs, psi0 = None, schedule = 'sin', options = opt):
    
    H0 = make_H0(H_i, H_p)
    H_floquet = qutip.QobjEvo([[H_i, ord3v2_l2.A_f], [H_p, ord3v2_l2.B_f]])
    
    # linking schedule
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    
    if psi0 is None:
        E0, psi0 = H0(t=0, args=Hargs).groundstate()
    fidelity, epsilon, hf = tools.prepare_hook(H0, Hargs)
    res = qutip.sesolve(H_floquet, psi0, time, e_ops=hf, args=Hargs, options=options, progress_bar=True)
    epsilon = np.array(epsilon).reshape(-1)
    
    return fidelity, epsilon

def return_ord3v2_l2(H_i, H_p, Hargs, schedule:str='sin'):
    H_floquet = qutip.QobjEvo([[H_i, ord3v2_l2.A_f], [H_p, ord3v2_l2.B_f]])
    Hargs = proto.link_schedule(Hargs=Hargs, schedule=schedule)
    return H_floquet, Hargs
