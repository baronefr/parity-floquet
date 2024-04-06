# %%

import lib.numcd as numcd
import lib.models as mods
import lib.protocols as proto

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model to test", type=str, default='lhz6m')
parser.add_argument("-schedule", help="lambda(t)", type=str, default='sin')
config = parser.parse_args()
print(config)

MODEL = config.model
SCHEDULE :str = config.schedule

# %% prepare schedule
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
elif MODEL == '3level':
    H_i, H_p = mods.model_3level()
else:
    raise ValueError('unknown model')



_, alphas_l1 = numcd.Floquet_l1_AB(H_i, H_p)
_, alphas_l2 = numcd.Floquet_l2_AB(H_i, H_p)



# %%
this_tau = 0.1
omega_0_factor : int = 4
omega_0 = omega_0_factor * 2*np.pi

# logger 
OUT_FILE :str = f'data/{MODEL}/schedule_tau-{this_tau}_w0{omega_0_factor}.npy'

    
# %% executing


P :int = 2000 # steps
time = np.linspace(0,this_tau,P+1)


print('  UA')
Hargs = {'tau':this_tau}
fidelity_UA, epsilon_UA = proto.ua.run_UA(H_i, H_p, time, Hargs, schedule=SCHEDULE)


#  CD l=1 -------------------------
print('  CD1')
Hargs = {'tau':this_tau, 'alpha_1':alphas_l1[0]}

fidelity_C1, epsilon_C1 = proto.cd.run_CD_l1(H_i, H_p, time, Hargs, schedule=SCHEDULE)



#  CD l=2 -------------------------
print('  CD2')
Hargs = {'tau':this_tau, 'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1]}

fidelity_C2, epsilon_C2 = proto.cd.run_CD_l2(H_i, H_p, time, Hargs, schedule=SCHEDULE)



# Floquet l=1 ---------------------
rateo = 2.5 * (10**2)
omega = rateo * omega_0    
omega_FE1 = omega

Hargs_l1 = {'tau':this_tau, 'omega_0':omega_0, 'omega':omega, 'rateo':rateo, 'P':P,
    'alpha_1':alphas_l1[0]
}

print('  l1')
fidelity_F1, epsilon_F1 = proto.claeys.run_Claeys_l1(H_i, H_p, time, Hargs_l1, schedule=SCHEDULE)



# Floquet l=2 ---------------------
rateo = 2.5 * (10**4)
omega = rateo * omega_0
omega_FE2 = omega

Hargs_l2 = {
    'tau':this_tau, 'omega_0':omega_0, 'omega':omega, 'rateo':rateo, 'P':P,
    'alpha_1':alphas_l2[0], 'alpha_2':alphas_l2[1]
}

print('  l2')
fidelity_F2, epsilon_F2 = proto.claeys.run_Claeys_l2(H_i, H_p, time, Hargs_l2, schedule=SCHEDULE)


data = {
    'eps_UA': np.array(epsilon_UA),
    'eps_CD1': np.array(epsilon_C1), 'eps_CD2':np.array(epsilon_C2),
    'eps_FE1': np.array(epsilon_F1), 'eps_FE2':np.array(epsilon_F2),
    'fid_UA': np.array(fidelity_UA),
    'fid_CD1': np.array(fidelity_C1), 'fid_CD2':np.array(fidelity_C2),
    'fid_FE1': np.array(fidelity_F1), 'fid_FE2':np.array(fidelity_F2),
    'omega_FE1':omega_FE1, 'omega_FE2':omega_FE2, 'omega0':omega_0
}

np.save(OUT_FILE,  data)  
