# %%

import csv
import numpy as np
import os

import lib.models as mods
import lib.protocols as proto
import lib.numcd as numcd




# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tau", help="total evolution time", type=float, default=0.1)
parser.add_argument("-model", help="model to test", type=str, default='lhz6m')
parser.add_argument("-protocol", help="protocol to test", type=str, default='claeys-l1')
parser.add_argument("-steps", help="steps of the annealing protocol", type=int, default=1000)
parser.add_argument("-schedule", help="lambda(t)", type=str, default='sin')
parser.add_argument("-parspace", help="parameter space to test", type=str, default='dense')
parser.add_argument("-doreference", help="compute reference values with UA and CD protocol", type=bool, default=True)
config = parser.parse_args()
print(config)


# %% parsing inputs + default parameters value

MODEL :str = config.model
PARSPACE :str = config.parspace
PROTOCOL :str = config.protocol
SCHEDULE :str = config.schedule
DO_REFERENCE :bool = config.doreference

tau :float = config.tau  # total evolution time

steps_reference :int = 200
steps_protocol :int = config.steps # can be 400
resolution_Teff :int = 50


# %% values to benchmark
if PARSPACE == 'dense':
    omega_0 = np.geomspace(0.345510729459222, 10, num=14*2-1-7)*2*np.pi
    omegas = 2*np.pi*np.arange(1,17)/tau
elif PARSPACE == 'dense-high':
    omega_0 = np.geomspace(0.345510729459222, 10, num=14*2-1-7)*2*np.pi
    omegas = 2*np.pi*np.arange(1,17)/tau
    omega_0 = omega_0[-3:]
elif PARSPACE == 'densefurther':
    omega_0 = np.geomspace(0.345510729459222, 10, num=14*2-1-7)*2*np.pi
    omegas = 2*np.pi*np.arange(60,160,5)/tau
elif PARSPACE == 'densefurther-high':
    omega_0 = np.geomspace(0.345510729459222, 10, num=14*2-1-7)*2*np.pi
    omegas = 2*np.pi*np.arange(60,160,5)/tau
    omega_0 = omega_0[-3:]
elif PARSPACE == 'minimal':
    omega_0 = np.geomspace(0.31622776601683794, 10, num=4*1+3)*2*np.pi
    omegas = 2*np.pi*np.arange(1,17)/tau
elif PARSPACE == 'extended':
    omega_0 = np.geomspace(0.31622776601683794, 100, num=4*2+3)*2*np.pi
    omegas = 2*np.pi*np.arange(1,21)/tau
else:
    raise ValueError('not valid parameter space!')



# %% load the matrices for a specific model

print('loading model:', MODEL)
if MODEL == 'lhz6m':
    H_i, H_p = mods.model_lhz6_medium()
elif MODEL == 'lhz6e':
    H_i, H_p = mods.model_lhz6_easy()
elif MODEL == 'lhz6h':
    H_i, H_p = mods.model_lhz6_hard()
elif MODEL == 'lhz6homo':
    H_i, H_p = mods.model_lhz6_homo()
elif MODEL == 'lhz10':
    H_i, H_p = mods.model_lhz10_medium()
elif MODEL == 'ising6':
    H_i, H_p = mods.model_ising(N=6)
elif MODEL == 'ising6c':
    H_i, H_p = mods.model_ising(N=6, x=-1, h=1, J=-1)
elif MODEL == 'ising12c':
    H_i, H_p = mods.model_ising(N=12, x=-1, h=1, J=-1)
elif MODEL == '3level':
    H_i, H_p = mods.model_3level()
else:
    raise ValueError('unknown model')



# %% set output file

OUT_FILE = f'data/{MODEL}/{PROTOCOL}_{SCHEDULE}/oscgr_tau-{tau}.csv'
print('bench file:', OUT_FILE)

# create directory if not existing
if not os.path.exists( os.path.dirname(OUT_FILE) ):
    print('creating path')
    os.makedirs( os.path.dirname(OUT_FILE) )

# write header, if not empty
do_write_header :bool = True
if os.path.exists(OUT_FILE):
    if os.stat(OUT_FILE).st_size > 0:
        do_write_header = False
if do_write_header:
    with open(OUT_FILE, 'w') as f:
        f.write("tau,P,omega,omega_0,fidelity,epsilon,Teff,intH,DintH,normax\n")

# define logger function
def logger(data):
    with open(OUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        #writer.writeheader()
        writer.writerow(data)




# %% init protocol rules

if PROTOCOL == 'claeys-l1':
    _, alphas_l1 = numcd.Floquet_l1_AB(H_i, H_p)
    Hargs = {'tau':tau, 'alpha_1':alphas_l1[0] }
    
    function_exe = proto.claeys.run_Claeys_l1
    function_return = proto.claeys.return_Claeys_l1
    function_Teff = proto.claeys.claeys_l1.effective_time_integrand
    function_CD = proto.cd.run_CD_l1

elif PROTOCOL == 'claeys-l2':
    _, alphas_l2 = numcd.Floquet_l2_AB(H_i, H_p)
    Hargs = { 'tau':tau, 'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1] }
    
    function_exe = proto.claeys.run_Claeys_l2
    function_return = proto.claeys.return_Claeys_l2
    function_Teff = proto.claeys.claeys_l2.effective_time_integrand
    function_CD = proto.cd.run_CD_l2

elif PROTOCOL == 'ord2-l1':
    _, alphas_l1 = numcd.Floquet_l1_AB(H_i, H_p)
    Hargs = {'tau':tau, 'alpha_1':alphas_l1[0]}
    
    function_exe = proto.exp.run_ord2_l1
    function_return = proto.exp.return_ord2_l1
    function_Teff = proto.exp.ord2_l1.effective_time_integrand
    function_CD = proto.cd.run_CD_l1

elif PROTOCOL == 'ord3-l1':
    _, alphas_l1 = numcd.Floquet_l1_AB(H_i, H_p)
    Hargs = {'tau':tau, 'alpha_1':alphas_l1[0]}
    
    function_exe = proto.exp.run_ord3_l1
    function_return = proto.exp.return_ord3_l1
    function_Teff = proto.exp.ord3_l1.effective_time_integrand
    function_CD = proto.cd.run_CD_l1

elif PROTOCOL == 'ord3v2-l1':
    _, alphas_l1 = numcd.Floquet_l1_AB(H_i, H_p)
    Hargs = {'tau':tau, 'alpha_1':alphas_l1[0]}
    
    function_exe = proto.exp.run_ord3v2_l1
    function_return = proto.exp.return_ord3v2_l1
    function_Teff = proto.exp.ord3v2_l1.effective_time_integrand
    function_CD = proto.cd.run_CD_l1

elif PROTOCOL == 'ord2-l2':
    _, alphas_l2 = numcd.Floquet_l2_AB(H_i, H_p)
    Hargs = { 'tau':tau, 'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1] }
    
    function_exe = proto.exp.run_ord2_l2
    function_return = proto.exp.return_ord2_l2
    function_Teff = proto.exp.ord2_l2.effective_time_integrand
    function_CD = proto.cd.run_CD_l2
    
elif PROTOCOL == 'ord3-l2':
    _, alphas_l2 = numcd.Floquet_l2_AB(H_i, H_p)
    Hargs = { 'tau':tau, 'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1] }
    
    function_exe = proto.exp.run_ord3_l2
    function_return = proto.exp.return_ord3_l2
    function_Teff = proto.exp.ord3_l2.effective_time_integrand
    function_CD = proto.cd.run_CD_l2

elif PROTOCOL == 'ord3v2-l2':
    _, alphas_l2 = numcd.Floquet_l2_AB(H_i, H_p)
    Hargs = { 'tau':tau, 'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1] }
    
    function_exe = proto.exp.run_ord3v2_l2
    function_return = proto.exp.return_ord3v2_l2
    function_Teff = proto.exp.ord3v2_l2.effective_time_integrand
    function_CD = proto.cd.run_CD_l2

else:
    raise ValueError('invalid protocol to test')





# %% computing reference UA and CD values (optional, for debugging purpose only)

if DO_REFERENCE:
    print('making UA')
    time_UA = np.linspace(0, tau, steps_reference)
    fidelity_UA, epsilon_UA = proto.ua.run_UA(H_i, H_p, time_UA, Hargs, schedule=SCHEDULE)

    print('making CD')
    time_CD = np.linspace(0, tau, steps_reference)
    fidelity_CD, epsilon_CD = function_CD(H_i, H_p, time_CD, Hargs, schedule=SCHEDULE)
    
    # write preamble reference values
    print('UA references:', fidelity_UA[-1], epsilon_UA[-1])
    print('CD references:', fidelity_CD[-1], epsilon_CD[-1])
    with open(OUT_FILE, 'a', newline='') as f:
        f.write(f"#REFERENCE:{fidelity_UA[-1]},{epsilon_UA[-1]},{fidelity_CD[-1]},{epsilon_CD[-1]}\n")



# %% do the real benchmark!

for omega_this in omegas:
    for omega_0_this in omega_0:
        print('exe omega, omega_0 = {}, {}'.format(omega_this, omega_0_this))

        time = np.linspace(0,tau,steps_protocol)
        P = time.shape[0]
        
        Hargs['omega_0'] = omega_0_this
        Hargs['omega'] = omega_this
        Hargs['rateo'] = omega_this/omega_0_this
        
        try:
            fidelity, epsilon = function_exe(H_i, H_p, time, Hargs, schedule=SCHEDULE)
        except KeyboardInterrupt:
            print('keyboard interrupt, closing')
            exit()
        except:
            fidelity = np.array([np.nan])
            epsilon = np.array([np.nan])

        # integrate norm
        try:
            HH, HH_args = function_return(H_i, H_p, Hargs, schedule=SCHEDULE)
            integral, ddintegral, normax = proto.norm_stats(HH, HH_args, max_samples=P)
        except:
            integral, ddintegral, normax = np.nan, np.nan, np.nan
        
        try:
            T_eff = proto.integrate_Teff(function_Teff, tau=tau, omega=omega_this, args=Hargs, schedule=SCHEDULE)
        except:
            T_eff = np.nan
        
        logger({'tau':tau, 'P':P, 
                'omega' : omega_this, 'omega_0' : omega_0_this, 
                'fidelity' : fidelity[-1], 'epsilon' : epsilon[-1], 
                'T_eff' : T_eff, 'intH':integral, 'DintH':ddintegral, 'normax':normax }
        )
