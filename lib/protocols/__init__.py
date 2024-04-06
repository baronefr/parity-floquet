
import numpy as np
import scipy

# inject available protocols
import lib.protocols.ua as ua
import lib.protocols.cd as cd
import lib.protocols.claeys as claeys
import lib.protocols.exp as exp




# %% [markdown]
# ## SCALE FUNCTIONS

class scale_sin:
    
    def lamb(t, tau):
        """Define the schedule function \lambda : [0,tau] -> [0,1]."""
        return np.sin( (np.pi/2)*(np.sin(np.pi*t/(2*tau))**2)  )**2

    def lamb_dot(t, tau):
        """Derivative of the scaling function \lambda."""
        b = np.pi*t/(2*tau)
        a = (np.pi/2)*(np.sin(b)**2)
        return np.sin(a)*np.cos(a)*np.sin(b)*np.cos(b)*(np.pi*np.pi/tau)

    def lamb_dotdot(t, tau):
        """Second derivative of the scaling function \lambda."""
        b = np.pi*t/(2*tau)
        c = (np.pi)*(np.sin(b)**2) # = 2a
        return np.cos(c)*np.sin(b)*np.cos(b)*np.sin(2*b)*((np.pi**4)/(4*(tau**2))) +\
            np.sin(c)*np.cos(2*b)*((np.pi**3)/(4*(tau**2)))
            

class scale_linear:
    def lamb(t, tau):
        """Define the schedule function \lambda : [0,tau] -> [0,1]."""
        return t/tau

    def lamb_dot(t, tau):
        """Derivative of the scaling function \lambda."""
        return np.ones_like(t)/tau

    def lamb_dotdot(t, tau):
        """Second derivative of the scaling function \lambda."""
        return np.ones_like(t)*0




# %%
# associate the scale functions classes to an alias
schedules_dict = {'sin':scale_sin, 'lin':scale_linear}

def link_schedule(Hargs:dict, schedule:str):
    if isinstance(schedule, str):
        schedule = schedules_dict[schedule]
    if schedule is not None:
        Hargs['scale_f'] = schedule.lamb
        Hargs['scale_f_dot'] = schedule.lamb_dot
        Hargs['scale_f_dotdot'] = schedule.lamb_dotdot
    return Hargs



# %% [markdown]
# ### extras

def integrate_Teff(function_Teff, tau, omega, args, schedule:str, resolution:int=50) -> float:
    # linking schedule
    args = link_schedule(Hargs=args, schedule=schedule)
    
    T_eff, deltaT_eff = scipy.integrate.quad(function_Teff, 0,tau, args=(args), limit=int( 1+(tau/(2*np.pi/omega)) )*resolution )
    return T_eff


def norm_stats(HH, HH_args:dict, max_samples:int):
    # compute the integral
    def integrand(t):
        return HH(t, args=HH_args).norm('fro')
    H_integral, delta_integral = scipy.integrate.quad(integrand, 0, HH_args['tau'])

    # compute the max value
    H_max = np.max([ HH(tt, args=HH_args).norm('fro') for tt in np.linspace(0,HH_args['tau'],max_samples) ])

    return H_integral, delta_integral, H_max