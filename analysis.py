# %%

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Polygon as mplPolygon
import matplotlib.patches as patches
from shapely.geometry import Polygon as shPolygon
from shapely.ops import unary_union


useLaTeX = True  # if True, use LaTeX backend for plots
if useLaTeX:
    plt.style.use('this.mplstyle')
    # patching for this analysis
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False



# %% select dataset & options for this run

tau :float = 0.1
MODEL :str = 'lhz6m'
# PROTOCOLS 'claeys-l1', 'ord2-l1', 'ord3-l1', 'ord3v2-l1'
PROTOCOL :str = 'claeys-l1'
SCHEDULE :str = 'sin'

DO_SAVEFIG = False
DO_TITLE = False

perc_tolerance = 0.02
CD_tolerance = 0.04 # limits in percentace of total range
UA_tolerance = 0.04

cbar_reflabel_shift = -1.3
cbar_marker_shift = 0.4
excess_marker_size_max = 40
excess_marker_size_min = 16
oscgr_plot_ratio = 0.14
cmap_extr_limiter = 0.05

OVERMAP_COLOR = '#303030'

# %% handling data directory

PREFIX = f'img/{MODEL}/{PROTOCOL}_{SCHEDULE}/'
if DO_SAVEFIG:
    if not os.path.exists( PREFIX ):
        os.makedirs( PREFIX )

# %% labels for plots

if PROTOCOL == 'claeys-l1':
    protocol_label = r"Claeys $\ell=1$"
elif PROTOCOL == 'claeys-l2':
    protocol_label = r"Claeys $\ell=2$"
elif PROTOCOL == 'ord2-l1':
    protocol_label = r"$\mathcal{O}$2 $\ell=1$"
elif PROTOCOL == 'ord3-l1':
    protocol_label = r"$\mathcal{O}$3 $\ell=1$"
elif PROTOCOL == 'ord3v2-l1':
    protocol_label = r"$\mathcal{O}$3L $\ell=1$"
else:
    protocol_label = PROTOCOL

model_label = MODEL.upper()

if SCHEDULE == 'sin':
    scale_label = r"$\lambda(t)=\lambda_{\sin}$"
elif SCHEDULE == 'lin':
    scale_label = r"$\lambda(t)=t/\tau$"

# %% retrieve UA T_eff

df_ua_bench = pd.read_csv(f"data/{MODEL}/UA-teff_{SCHEDULE}.csv")
cutoff = df_ua_bench[ df_ua_bench['epsilon'] > 0 ]['epsilon'].min()
df_ua_bench.loc[ df_ua_bench['epsilon'] < cutoff, 'epsilon'] = cutoff
assert np.all(np.diff(df_ua_bench['tau']) > 0), 'must be monotone'

def matched_epsilon_from_Teff(teff):
    # np.interp args: x, xp (increasing), fp
    return np.interp(np.float64(teff), np.float64(df_ua_bench['tau']), np.float64(df_ua_bench['epsilon']))

def matched_epsilon_from_intH(intH):
    return np.interp(np.float64(intH), np.float64(df_ua_bench['intH']), np.float64(df_ua_bench['epsilon']))

x = np.geomspace(min(df_ua_bench['tau']), max(df_ua_bench['tau']), 100)
plt.plot(df_ua_bench['tau'], 1-df_ua_bench['epsilon'])
plt.plot(x, 1 - matched_epsilon_from_Teff(x), '--' )
plt.xscale('log')
plt.ylabel(r"$1-\varepsilon$")
plt.xlabel(r"$\tau_{\text{eff}}$")
# NOTE: check overlap!


# %%

plt.plot(df_ua_bench['tau'], df_ua_bench['intH'])
plt.xscale('log')
plt.ylabel(r"$\int \|H\|$")
plt.xlabel(r"$\tau_{\text{eff}}$")


# %% functions to process data

def get_grid(df:pd.core.frame.DataFrame, key_1:str, key_2:str, target:str):
    """
    Return the matrix of fidelities gven the dataframe collecting all data.

    Args:
        df (pd.core.frame.DataFrame): dataframe
        key_1 (str): First key for data point
        key_2 (str): Second key for data point
        target (str): data to be extracted
    """
    val_1 = sorted( df[key_1].unique() )
    val_2 = sorted( df[key_2].unique() )

    matrix = np.empty(shape=(len(val_1),len(val_2)))
    matrix[:] = np.nan
    rateo = np.empty(shape=(len(val_1),len(val_2)))
    rateo[:] = np.nan
    for index, row in df.iterrows():
        x, y = row[key_1], row[key_2]
        matrix[val_1.index(x), val_2.index(y)] = row[target]
    return matrix[:,:], np.array(val_1), np.array(val_2)


def process_df(df):
    df['r'] = 1 - df['epsilon']
    df['rateo'] = np.round(df['omega']/df['omega_0'], 4)
    
    # equivalent epsilon from T_eff
    df['eps_teff'] = matched_epsilon_from_Teff(df['Teff'])
    df['eps_intH'] = matched_epsilon_from_intH(df['intH'])
    return df


# %% aesthetics functions


def format_func(value, tick_number) -> str:
    """custom formatter function to format the colorbar labels"""
    # format the value as a decimal with a specified number of decimal places
    return "{:.3f}".format(value)


def get_cmap_center(omap, center_color, fraction) -> colors.ListedColormap:
    """Create a colormap with a new color around 0."""
    if isinstance(center_color, str):
        center_color = colors.to_rgba(center_color)
    samples = 1000
    fraction = fraction/2
    original_cmap = cm.get_cmap(omap, 1000)
    newcolors = original_cmap(np.linspace(0, 1, samples))
    newcolors[int(500-(fraction*1000)):int(500+(fraction*1000)), :] = center_color
    return colors.ListedColormap(newcolors)


def plt_titling(pl) -> None:
    """Set titling of a plot, default style."""
    if DO_TITLE:
        pl.title(f"{model_label}", loc='left', fontsize = 13)
        pl.title(f"{protocol_label}", loc='right', fontsize = 11) 
        # NOTE: scale label  f"\n{scale_label}"  has been removed


def crop_map(base:str, lo:float, hi:float, grey_value:int = 24, augm:float = 0.0):
    if isinstance(base,str):
        targetcmap = cm.get_cmap(base, 256)
    else:
        targetcmap = base
    grey = np.array([grey_value/256, grey_value/256, grey_value/256, 1])
    middle = targetcmap(np.linspace(0, 1, 256))
    low = np.empty( (int((lo-augm)*256),4) )
    high = np.empty( (int((hi-augm)*256),4) )
    low[:,:]=grey
    high[:,:]=grey
    newcolors = np.concatenate( (low,middle,high) )
    return colors.ListedColormap(newcolors)





# %% [markdown]
# ### analysis on oscillations


# set new properties of plt
plt.rcParams["ytick.minor.visible"] = False

# load data
DATA_FILE = f'data/{MODEL}/{PROTOCOL}_{SCHEDULE}/oscgr_tau-{tau}.csv'
df = pd.read_csv(DATA_FILE, comment='#')
df = process_df(df)

df['gain'] = (1-df['epsilon']) - (1-df['eps_teff'])
df['osc'] = np.rint((df['omega']*tau)/(2*np.pi)).astype(int)
df['cerr'] = df['intH']/df['osc']
df['Teffosc'] = df['Teff']/df['osc']

try:
    reference = None
    with open(DATA_FILE,"r") as file:
        for line in file:
            if line.startswith("#REFERENCE:"):
                reference = line.rsplit()[0]
                break
    reference = reference.replace("#REFERENCE:", "").split(',')
    ref_fidelity_UA, ref_epsilon_UA = float(reference[0]), float(reference[1])
    ref_fidelity_CD, ref_epsilon_CD = float(reference[2]), float(reference[3])
except:
    ref_fidelity_UA, ref_epsilon_UA = np.nan, np.nan
    ref_fidelity_CD, ref_epsilon_CD = np.nan, np.nan

print('reference values:  UA', ref_fidelity_UA, ref_epsilon_CD)
print('                   CD', ref_fidelity_CD, ref_epsilon_CD)

ref_R_CD = (1-ref_epsilon_CD)
ref_R_UA = (1-ref_epsilon_UA)





# %% [markdown]
# ### plot distribution of samples

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_ua_bench['tau'], 1-df_ua_bench['epsilon'], 'k', lw=2)
ax.text(50, 1-matched_epsilon_from_Teff(50)-0.07, 'UA')

cmap = cm.get_cmap('jet_r')
group = df.groupby('osc')
for gn, dfg in group:
    ax.scatter(dfg['Teff'], 1-dfg['epsilon'], alpha=0.7,
        c=[cmap((gn-1)/len(group))]  )

thr = min(1-df_ua_bench['epsilon'])
ax.text(0.15, thr+0.05, '$\mathcal{G}>0$', c='green')
ax.text(0.15, thr-0.10, '$\mathcal{G}<0$', c='grey')

oscmin = min(df['osc'])
oscmax = max(df['osc'])
norm = colors.Normalize(vmin=oscmin, vmax=oscmax)

divider = make_axes_locatable(ax)
if MODEL == '3level':
    locs = ax.inset_axes((0.55, 0.3, 0.4, 0.04))
else:
    locs = ax.inset_axes((0.05, 0.9, 0.4, 0.04))

cb = fig.colorbar( cm.ScalarMappable(norm=norm, cmap=cmap), 
    cax=locs, 
    orientation='horizontal', label=r'$\omega\tau/2\pi$',
    ticks = np.linspace(oscmin,oscmax,4)
)
plt.xlim((0.1,100))
plt.xscale('log')
plt.ylabel(r"$\mathcal{M}$", rotation=0,  labelpad=10)
plt.xlabel(r"$\tau_{\text{eff}}$")
plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_projection.pdf', bbox_inches='tight')




# %% [markdown]
# ### plot distribution of samples (shaded)

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_ua_bench['tau'], 1-df_ua_bench['epsilon'], 'k', lw=2)
ax.text(50, 1-matched_epsilon_from_Teff(50)-0.07, 'UA')

cmap = cm.get_cmap('jet_r')
group = df.groupby('osc')
for gn, dfg in group:
    this_data = 1-dfg['epsilon']
    this_color = np.array(cmap((gn-1)/len(group)))
    this_color_shaded = this_color*0.5
    this_color_shaded[-1] = 1
    this_colors = [this_color if cond else this_color_shaded for cond in np.array( dfg['gain'] > 0 ) ] 
    ax.scatter(dfg['Teff'], this_data, alpha=0.7, c=this_colors  )

thr = min(1-df_ua_bench['epsilon'])
ax.text(0.15, thr+0.05, '$\mathcal{G}>0$', c='green')
ax.text(0.15, thr-0.10, '$\mathcal{G}<0$', c='grey')

oscmin = min(df['osc'])
oscmax = max(df['osc'])
norm = colors.Normalize(vmin=oscmin, vmax=oscmax)

divider = make_axes_locatable(ax)
if MODEL == '3level':
    locs = ax.inset_axes((0.55, 0.3, 0.4, 0.04))
else:
    locs = ax.inset_axes((0.05, 0.9, 0.4, 0.04))

cb = fig.colorbar( cm.ScalarMappable(norm=norm, cmap=cmap), 
    cax=locs, 
    orientation='horizontal', label=r'$\omega\tau/2\pi$',
    ticks = np.linspace(oscmin,oscmax,4)
)
plt.xlim((0.1,100))
plt.xscale('log')
plt.ylabel(r"$\mathcal{M}$", rotation=0,  labelpad=10)
plt.xlabel(r"$\tau_{\text{eff}}$")
plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_projection-shaded.pdf', bbox_inches='tight')



# %% plot NORM CYCLE ERROR

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'cerr')
omega_0 = omega_0/(2*np.pi)
print(f'nrm min={np.min(matrix)}, avg={np.average(matrix)}')
X,Y=np.meshgrid(omega_0,osc)
im = plt.pcolor(X, Y, 2*matrix, cmap='Greys')

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\langle\delta\rangle$")

plt_titling(plt)
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_cycle-error.pdf', bbox_inches='tight')




# %% plot fidelity

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'fidelity')
omega_0 = omega_0/(2*np.pi)

X,Y=np.meshgrid(omega_0,osc)
im = plt.pcolor(X, Y, matrix, cmap='plasma')

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathcal{F}$", rotation=0,  labelpad=10)

cbar.ax.scatter(cbar_marker_shift, ref_fidelity_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_fidelity_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_UA, 'UA', fontsize=8, verticalalignment='center')

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_fidelity.pdf', bbox_inches='tight')



# %% plot energy merit

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'r')
omega_0 = omega_0/(2*np.pi)

X,Y=np.meshgrid(omega_0,osc)
im = plt.pcolor(X, Y, matrix, cmap='viridis')

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathcal{M}$", rotation=0,  labelpad=10)

cbar.ax.scatter(cbar_marker_shift, ref_R_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_R_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_R_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_R_UA, 'UA', fontsize=8, verticalalignment='center')

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_M.pdf', bbox_inches='tight')






# %% plot fidelity (cropped)

cmap = plt.get_cmap('plasma').copy()
cmap.set_extremes(under=OVERMAP_COLOR, over=OVERMAP_COLOR)

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'fidelity')
omega_0 = omega_0/(2*np.pi)

upper_limit = ref_fidelity_CD + (ref_fidelity_CD-ref_fidelity_UA)*CD_tolerance
lower_limit = ref_fidelity_UA - (ref_fidelity_CD-ref_fidelity_UA)*UA_tolerance

X,Y=np.meshgrid(omega_0,osc)
im = plt.pcolor(X, Y, matrix, vmax=upper_limit, vmin=lower_limit, cmap=cmap)

mark_x, mark_y = np.where( matrix > upper_limit )
values = matrix[ np.where( matrix > upper_limit ) ] - upper_limit
if len(values) > 0:
    values = values/np.max(values)
    plt.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min )
mark_x, mark_y = np.where( matrix < lower_limit )
values = lower_limit - matrix[ np.where( matrix < lower_limit ) ]
if len(values) > 0:
    values = values/np.max(values)
    plt.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min )


plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im, extend='both', extendfrac=0.05)
cbar.set_label(r"$\mathcal{F}$", rotation=0,  labelpad=10)

cbar.ax.scatter(cbar_marker_shift, ref_fidelity_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_fidelity_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_UA, 'UA', fontsize=8, verticalalignment='center')

cbar.ax.text(0.500, upper_limit+0.0012, '+', fontsize=10, c='w', verticalalignment='center', horizontalalignment='center')
cbar.ax.text(0.500, lower_limit-0.0012, '-', fontsize=14, c='w', verticalalignment='center', horizontalalignment='center')

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_fidelity-crop.pdf', bbox_inches='tight')




# %% plot energy merit (cropped)

cmap = plt.get_cmap('viridis').copy()
cmap.set_extremes(under=OVERMAP_COLOR, over=OVERMAP_COLOR)

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'r')
omega_0 = omega_0/(2*np.pi)

upper_limit = ref_R_CD + (ref_R_CD-ref_R_UA)*CD_tolerance
lower_limit = ref_R_UA - (ref_R_CD-ref_R_UA)*UA_tolerance

X,Y = np.meshgrid(omega_0,osc)
im = plt.pcolor(X,Y,matrix, vmax=upper_limit, vmin=lower_limit, cmap=cmap)

mark_x, mark_y = np.where( matrix > upper_limit )
values = matrix[ np.where( matrix > upper_limit ) ] - upper_limit
if len(values) > 0:
    values = values/np.max(values)
    plt.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min)
mark_x, mark_y = np.where( matrix < lower_limit )
values = lower_limit - matrix[ np.where( matrix < lower_limit ) ]
if len(values) > 0:
    values = values/np.max(values)
    plt.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min)

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im, extend='both', extendfrac=0.04)
cbar.set_label(r"$\mathcal{M}$", rotation=0,  labelpad=10)

cbar.ax.scatter(cbar_marker_shift, ref_R_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_R_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_R_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_R_UA, 'UA', fontsize=8, verticalalignment='center')

cbar.ax.text(0.490, upper_limit+0.003, '+', fontsize=8, c='w', verticalalignment='center', horizontalalignment='center')
cbar.ax.text(0.490, lower_limit-0.003, '-', fontsize=8, c='w', verticalalignment='center', horizontalalignment='center')


plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_M-crop.pdf', bbox_inches='tight')




# %% plot R - R_CD

def fmt_label(x,p):
    s = f"{100*x:+.0f}"
    if s.endswith("0"):
        s = f"{100*x:.0f}"
    return rf"${s}\%$" if plt.rcParams["text.usetex"] else f"{s} %"

# custom colormap
colr = ['#3b0762','w', '#f24500']
positions = [0.0, 0.5, 1.0]
cmap = colors.LinearSegmentedColormap.from_list('orange_white_purple', list(zip(positions, colr)))
cmap.set_extremes(under=OVERMAP_COLOR, over=OVERMAP_COLOR)

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'r')
omega_0 = omega_0/(2*np.pi)

X,Y = np.meshgrid(omega_0,osc)
data = matrix-ref_R_CD
if cmap_extr_limiter is None:
    cmap_extr = np.max( np.abs(data) )
else:
    cmap_extr = cmap_extr_limiter

im = plt.pcolor(X,Y,data, vmax=cmap_extr, vmin=-cmap_extr, cmap=cmap)

outsider_count = 0
if cmap_extr_limiter is not None:
    upper_limit =  cmap_extr
    lower_limit = -cmap_extr
    mark_x, mark_y = np.where( data > upper_limit )
    values = data[ np.where( data > upper_limit ) ] - upper_limit
    if len(values) > 0:
        outsider_count += len(values)
        values = values/np.max(values)
        plt.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min)
    mark_x, mark_y = np.where( data < lower_limit )
    values = lower_limit - data[ np.where( data < lower_limit ) ]
    if len(values) > 0:
        outsider_count += len(values)
        values = values/np.max(values)
        plt.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min)

print('elements inside:', data.shape[0] * data.shape[1] - outsider_count)


plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im, extend='both', extendfrac=0.05, format=fmt_label)
cbar.set_label(r"$\mathcal{M}-\mathcal{M}_{\mathrm{CD}}$")
if cmap_extr_limiter is not None:
    #cbar.ax.text(2, cmap_extr_limiter*1.1, "$(\pm" + str(cmap_extr_limiter) + ")$", size= 'small' )
    cbar.ax.text(-1, -cmap_extr_limiter*1.5, "$(\mathcal{M}_{\mathrm{CD}} =" + str(np.round(ref_R_CD,2)) + ")$", size= 'small' )
    cbar.ax.text(1.3, upper_limit*1.07, f'$>+{int(cmap_extr_limiter*100)}\%$', fontsize=10, c='k', verticalalignment='center', horizontalalignment='left')
    cbar.ax.text(1.3, lower_limit*1.07, f'$<-{int(cmap_extr_limiter*100)}\%$', fontsize=10, c='k', verticalalignment='center', horizontalalignment='left')

if cmap_extr_limiter is not None:
    cbar.ax.text(0.4950, upper_limit+0.002, '+', fontsize=10, c='w', verticalalignment='center', horizontalalignment='center')
    cbar.ax.text(0.4950, lower_limit-0.002, '-', fontsize=14, c='w', verticalalignment='center', horizontalalignment='center')

plt_titling(plt)
#plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_MvsCD.pdf', bbox_inches='tight')




# %% slice

dfs = df[ df['osc'] == 10 ]

fig, ax1 = plt.subplots(figsize=(5,3))

ax1.plot(dfs['omega_0']/(2*np.pi), dfs['Teff'], label=r'$\tau_{\text{eff}}$', c='#9B0014', marker='s')
ax1.set_ylabel(r'$\tau_{\text{eff}}$', c='#9B0014', rotation=0,  labelpad=10)

ax2 = ax1.twinx()

#plt.plot(dfs['omega_0']/(2*np.pi), dfs['Teff'], label='Teff')
ax2.plot(dfs['omega_0']/(2*np.pi), 1-dfs['epsilon'] - ref_R_CD,
         c="#083966", marker='d', label=r"$\mathcal{M}-\mathcal{M}_{\mathrm{CD}}$")
ax2.set_ylabel(r'$\mathcal{M}-\mathcal{M}_{\mathrm{CD}}$', c="#083966")
ax2.axhline(y=0, xmin=0.1/(2*np.pi), xmax=10/(2*np.pi), c='k', linestyle='--')
ax2.text(4.5,0.007,r'$\mathcal{M}_{\mathrm{CD}}='+f"{np.round(ref_R_CD,2)}$", fontsize=10)
ax2.text(1,0.05,r'fixing $\omega\tau/2\pi=10$', fontsize=10)


ax1.set_xlabel(r"$\omega_0/2\pi$")
plt.xscale('log')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_MvsCD-sliced.pdf', bbox_inches='tight')


# %% slice (separated)

dfs = df[ df['osc'] == 10 ]

fig, ax = plt.subplots(figsize=(5,5), nrows=2, sharex=True)

ax[0].plot(dfs['omega_0']/(2*np.pi), dfs['Teff'], label=r'$\tau_{\text{eff}}$', c='#9B0014', marker='s')
ax[0].set_ylabel(r'$\tau_{\text{eff}}$', c='#9B0014', rotation=0,  labelpad=10)

#plt.plot(dfs['omega_0']/(2*np.pi), dfs['Teff'], label='Teff')
ax[1].plot(dfs['omega_0']/(2*np.pi), 1-dfs['epsilon'] - ref_R_CD,
         c="#083966", marker='d', label=r"$\mathcal{M}-\mathcal{M}_{CD}$")
ax[1].set_ylabel(r'$\mathcal{M}-\mathcal{M}_{\mathrm{CD}}$', c="#083966")
ax[1].axhline(y=0, xmin=0.1/(2*np.pi), xmax=10/(2*np.pi), c='k', linestyle='--')
ax[1].text(4.5,0.007,r'$\mathcal{M}_{\mathrm{CD}}='+f"{np.round(ref_R_CD,2)}$", fontsize=10)
#ax[1].text(1,0.05,r'fixing $\omega\tau/2\pi=10$', fontsize=10)


ax[1].set_xlabel(r"$\omega_0/2\pi$")
plt.xscale('log')
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_MvsCD-sliced-separate.pdf', bbox_inches='tight')




# %% plot R - R_UA

# custom colormap
colr = ['midnightblue','w', 'deeppink']
positions = [0.0, 0.5, 1.0]
cmap = colors.LinearSegmentedColormap.from_list('orange_white_purple', list(zip(positions, colr)))
cmap = 'RdBu_r'

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'r')
omega_0 = omega_0/(2*np.pi)

X,Y = np.meshgrid(omega_0,osc)
data = matrix-ref_R_UA
cmap_extr = np.max( np.abs(data) )
im = plt.pcolor(X,Y,matrix-ref_R_UA, vmax=cmap_extr, vmin=-cmap_extr, cmap=cmap)

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathcal{M}-\mathcal{M}_{\mathrm{UA}}$")

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_MvsUA.pdf', bbox_inches='tight')








# %% plot T_eff

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'Teff')
omega_0 = omega_0/(2*np.pi)

X,Y = np.meshgrid(omega_0,osc)
im = plt.pcolor(X,Y,matrix)

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\tau_{\text{eff}}$")

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_taueff.pdf', bbox_inches='tight')








# %% gain 

def get_contour(matrix, omega_0, omega, threshold):
    Y,X = np.where(matrix >= threshold)
    omega_0 = np.concatenate( (np.array([0.29]), omega_0, np.array([11])) ) # WARN: manual correction!!
    midway_x = np.sqrt(omega_0[1:]*omega_0[:-1])
    positions = np.dstack((midway_x[X], omega[Y]))[0]

    def get_Xcorrection(xy):
        val = np.diff(midway_x)
        return val[ np.argmax( xy[0] < omega_0 ) -1 ]

    # Create a rectangle per position and merge them.
    rectangles = [shPolygon([xy+[0,-0.5], xy + [get_Xcorrection(xy), -0.5], xy + [get_Xcorrection(xy), 0.5], xy + [0, 0.5]]) for xy in positions ]
    polygons = unary_union(rectangles)

    # Shapely will return either a Polygon or a MultiPolygon. 
    # Make sure the structure is the same in any case.
    if polygons.geom_type == "Polygon":
        polygons = [polygons]
    else:
        polygons = polygons.geoms
    return polygons


df['gain'] = (1-df['epsilon']) - (1-df['eps_teff'])

fig, ax = plt.subplots()

matrix, omega, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'gain')
omega_0 = omega_0/(2*np.pi)

scale_value = np.nanmax( np.abs(matrix) )
scale_value = 0.5
retain_value = 0.03
#newcmap = get_cmap_center('bwr_r', 'gold', retain_value/(scale_value))
newcmap = cm.get_cmap('RdBu_r', 1000)
# selecting one percent to be 

X,Y = np.meshgrid(omega_0,omega)
im = ax.pcolor(X,Y,matrix, cmap=newcmap, vmin=-scale_value, vmax=scale_value)

# Add the matplotlib Polygon patches
threshold = 0
for polygon in get_contour(matrix, omega_0, omega, threshold):
    ax.add_patch(mplPolygon(polygon.exterior.coords, 
        fc='none', ec='k', lw=1.7, linestyle='--', clip_on=False))

print('number of positive gains:', np.sum(matrix > 0) )

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathcal{G}$", rotation=0, labelpad=10)

cbar.ax.text(-1.3,0.23, f"$\ge{int(threshold*100)}\%$", 
    size='x-small', color='k', rotation='vertical', va='center')
rect = patches.Rectangle((-0.15, threshold), 1.15, scale_value-2*threshold, 
    linewidth=1.5, edgecolor='k', linestyle='--',
    facecolor='none', clip_on=False, zorder=10)
cbar.ax.add_patch(rect)

plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_gain.pdf', bbox_inches='tight')

print('max gain:', np.max(matrix))



# %% gain from  integral||H||



df['gain_intH'] = (1-df['epsilon']) - (1-df['eps_intH'])

fig, ax = plt.subplots()

matrix, omega, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'gain_intH')
omega_0 = omega_0/(2*np.pi)

scale_value = np.nanmax( np.abs(matrix) )
scale_value = 0.5
retain_value = 0.03

X,Y = np.meshgrid(omega_0,omega)
im = ax.pcolor(X,Y,matrix, cmap=newcmap, vmin=-scale_value, vmax=scale_value)

# Add the matplotlib Polygon patches
for polygon in get_contour(matrix, omega_0, omega, threshold):
    ax.add_patch(mplPolygon(polygon.exterior.coords, 
        fc='none', ec='k', lw=1.7, linestyle='--', clip_on=False))

print('number of positive gains:', np.sum(matrix > 0) )

plt.ylabel(r"$\omega\tau/2\pi$")
plt.xlabel(r"$\omega_0/2\pi$")

plt.gca().set_aspect(oscgr_plot_ratio)
plt.xscale('log')

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathcal{G}'$", rotation=0)

cbar.ax.text(-1.3,0.23, f"$\ge{int(threshold*100)}\%$", 
    size='x-small', color='k', rotation='vertical', va='center')
rect = patches.Rectangle((-0.15, threshold), 1.35, scale_value-2*threshold, 
    linewidth=1.5, edgecolor='k', linestyle='--',
    facecolor='none', clip_on=False, zorder=10)
cbar.ax.add_patch(rect)


plt_titling(plt)
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_gain-NORM.pdf', bbox_inches='tight')

# %%
