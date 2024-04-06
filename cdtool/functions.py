
import numpy as np

class scale_sin:

    def f(t, tau):
        """Define the schedule function \lambda : [0,tau] -> [0,1]."""
        return np.sin( (np.pi/2)*(np.sin(np.pi*t/(2*tau))**2)  )**2

    def fdot(t, tau):
        """Derivative of the scaling function \lambda."""
        b = np.pi*t/(2*tau)
        a = (np.pi/2)*(np.sin(b)**2)
        return np.sin(a)*np.cos(a)*np.sin(b)*np.cos(b)*(np.pi*np.pi/tau)



class pulse_sin3:

    def f(t : float, tau : float, beta : np.ndarray):
        """control function f(t, beta) to use for qoc"""
        k = np.arange(len(beta)) + 1
        
        if isinstance(t, np.ndarray):
            # fix the sum broadcast when time is an array
            t = t.reshape(-1, 1)
            return np.sum( (beta)*np.sin( np.pi*k*t/tau )*np.sin( np.pi*t/tau ), axis=1 )
        else:
            return np.sum( (beta)*np.sin( np.pi*k*t/tau )*np.sin( np.pi*t/tau ) )

    def fdot(t : float, tau : float, beta : np.ndarray):
        """control function f(t, beta) derivative to use for qoc"""
        k = np.arange(len(beta)) + 1
        
        if isinstance(t, np.ndarray):
            # fix the sum broadcast when time is an array
            t = t.reshape(-1, 1) # TODO: check correctness
            return np.sum( np.sin( np.pi*t/tau )*(beta)*np.cos( np.pi*k*t/tau )*(np.pi*k/tau), axis=1 ) +\
                np.sum( np.cos( np.pi*t/tau )*(beta)*np.sin( np.pi*k*t/tau )*(np.pi/tau), axis=1 )
        else:
            return np.sum( np.sin( np.pi*t/tau )*(beta)*np.cos( np.pi*k*t/tau )*(np.pi*k/tau) ) +\
                np.sum( np.cos( np.pi*t/tau )*(beta)*np.sin( np.pi*k*t/tau )*(np.pi/tau))

class pulse_sin:

    def f(t : float, tau : float, beta : np.ndarray):
        """control function f(t, beta) to use for qoc"""
        k = np.arange(len(beta)) + 1
        
        if isinstance(t, np.ndarray):
            # fix the sum broadcast when time is an array
            t = t.reshape(-1, 1)
            return np.sum( (beta)*np.sin( np.pi*k*t/tau ), axis=1 )
        else:
            return np.sum( (beta)*np.sin( np.pi*k*t/tau ) )

    def fdot(t : float, tau : float, beta : np.ndarray):
        """control function f(t, beta) derivative to use for qoc"""
        k = np.arange(len(beta)) + 1
        
        if isinstance(t, np.ndarray):
            # fix the sum broadcast when time is an array
            t = t.reshape(-1, 1)
            return np.sum( (beta)*np.cos( np.pi*k*t/tau )*(np.pi*k/tau), axis=1 )
        else:
            return np.sum( (beta)*np.cos( np.pi*k*t/tau )*(np.pi*k/tau) )



# TODO: get rid of this
def parametrized_gaussian_pulse(x, c, sigma, tau, ampl=1):
    return ampl*np.exp(-((x-c*tau) / sigma)**2)

def boundary_filter(x, coeff = 100, tau=1):
    return (1-np.exp(- (coeff * x)**2) -np.exp(-(coeff * (x-tau))**2))