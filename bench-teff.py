# %%

import lib.protocols as proto
import lib.models as mods

import numpy as np
import csv
import os



# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="model to test", type=str, default='lhz6m')
parser.add_argument("-schedule", help="lambda(t)", type=str, default='sin')
parser.add_argument("-samples", help="number of tau samples", type=int, default=1001)
parser.add_argument("-steps", help="number of steps per simulation", type=int, default=400)
config = parser.parse_args()
print(config)

samples = config.samples
steps = config.steps
MODEL = config.model
SCHEDULE = config.schedule

# %% prepare system
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


# %%

anneal_times = np.geomspace(0.01,1e3,samples)





# %% logger 

OUT_FILE = f'data/{MODEL}/UA-teff_{SCHEDULE}.csv'

# create directory if not existing
if not os.path.exists( os.path.dirname(OUT_FILE) ):
    os.makedirs( os.path.dirname(OUT_FILE) )

# write header, if not empty
do_write_header :bool = True
if os.path.exists(OUT_FILE):
    if os.stat(OUT_FILE).st_size > 0:
        do_write_header = False
if do_write_header:
    with open(OUT_FILE, 'w') as f:
        f.write("tau,intH,DintH,normax,epsilon,fidelity\n")

def logger(data):
    with open(OUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        #writer.writeheader()
        writer.writerow(data)

    
    
# %%

for tau in anneal_times:
    print(f'>> exe tau = {tau}')
    
    Hargs = {'tau': tau}
    time = np.linspace(0,tau,steps+1)
    fidelity_UA, epsilon_UA = proto.ua.run_UA(H_i, H_p, time, Hargs, schedule=SCHEDULE)
    
    HH, HH_args = proto.ua.return_UA(H_i, H_p, Hargs, schedule=SCHEDULE)
    integral, ddintegral, normax = proto.norm_stats(HH, HH_args,steps)

    data_ = { 
        'tau':tau, 'intH':integral, 'DintH':ddintegral, 'normax':normax,
        'epsilon':epsilon_UA[-1], 'fidelity':fidelity_UA[-1]
    }
    logger(data_)


