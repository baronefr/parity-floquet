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
# PROTOCOLS 'claeys-l2', 'ord3-l2', ord3v2-l2
PROTOCOL :str = 'claeys-l2'
SCHEDULE :str = 'sin'

DO_SAVEFIG = True
DO_TITLE = False

perc_tolerance = 0.02
CD_tolerance = 0.04 # limits in percentace of total range
UA_tolerance = 0.04

cbar_reflabel_shift = -1.3
cbar_marker_shift = 0.4
excess_marker_size_max = 40
excess_marker_size_min = 12
oscgr_plot_ratio = 0.14
cmap_extr_limiter = 0.05

OVERMAP_COLOR = '#303030'

# %% handling data directory

PREFIX = f'img/{MODEL}/{PROTOCOL}_{SCHEDULE}/'
if DO_SAVEFIG:
    if not os.path.exists( PREFIX ):
        os.makedirs( PREFIX )

# %% labels for plots

if PROTOCOL == 'claeys-l2':
    protocol_label = r"Claeys $\ell=2$"
elif PROTOCOL == 'ord2-l2':
    protocol_label = r"$\mathcal{O}$2 $\ell=2$"
elif PROTOCOL == 'ord3-l2':
    protocol_label = r"$\mathcal{O}$3 $\ell=2$"
elif PROTOCOL == 'ord3v2-l2':
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
    return np.interp(np.float64(teff), np.float64(df_ua_bench['tau']), np.float64(df_ua_bench['epsilon']), right=0)

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


def plt_titling(pl, suptitle:bool=None) -> None:
    """Set titling of a plot, default style."""
    if DO_TITLE:
        str_L = f"{model_label}" + f"\n{protocol_label}"
        str_R = f"$\\tau={tau}$" #+ f"\n{scale_label}"

        if suptitle is not None:
            plt.text(.13, 0.91, str_L, transform=fig.transFigure, horizontalalignment='left', fontsize = 13)
            plt.text(.71, 0.91, str_R, transform=fig.transFigure, horizontalalignment='right', fontsize = 11)
        else:
            pl.title(str_L, loc='left', fontsize = 13)
            pl.title(str_R, loc='right', fontsize = 11)

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
cutoff = 2000

df['gain'] = (1-df['epsilon']) - (1-df['eps_teff'])
df['osc'] = np.rint((df['omega']*tau)/(2*np.pi)).astype(int)
df['cerr'] = df['intH']/df['osc']
df['Teffosc'] = df['Teff']/df['osc']

df_low = df[ df['omega'] < cutoff ]
df_up = df[ df['omega'] > cutoff ]

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
ax.hlines(1-min(df_ua_bench['epsilon']), 1000,10000, 'k', lw=2)
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
    locs = ax.inset_axes((0.05, 0.9, 0.3, 0.04))
cb = fig.colorbar( cm.ScalarMappable(norm=norm, cmap=cmap), 
    cax=locs, 
    orientation='horizontal', label=r'$\omega\tau/2\pi$',
    ticks = np.linspace(oscmin,oscmax,3)
)
plt_titling(plt)
plt.xlim((0.1,5000))
plt.xscale('log')
plt.ylabel(r"$\mathcal{M}$", rotation=0, labelpad=10)
plt.xlabel(r"$\tau_{\text{eff}}$")
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_projection.pdf', bbox_inches='tight')



# %% [markdown]
# ### plot distribution of samples (shaded)

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_ua_bench['tau'], 1-df_ua_bench['epsilon'], 'k', lw=2)
ax.hlines(1-min(df_ua_bench['epsilon']), 1000,10000, 'k', lw=2)
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
    locs = ax.inset_axes((0.05, 0.9, 0.3, 0.04))
cb = fig.colorbar( cm.ScalarMappable(norm=norm, cmap=cmap), 
    cax=locs, 
    orientation='horizontal', label=r'$\omega\tau/2\pi$',
    ticks = np.linspace(oscmin,oscmax,3)
)
plt_titling(plt)
plt.xlim((0.1,5000))
plt.xscale('log')
plt.ylabel(r"$\mathcal{M}$", rotation=0, labelpad=10)
plt.xlabel(r"$\tau_{\text{eff}}$")
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_projection-shaded.pdf', bbox_inches='tight')



# %% [markdown]
# ## fidelity

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'fidelity')
omega_0 = omega_0/(2*np.pi)
X,Y=np.meshgrid(omega_0,osc)

fig, axs = plt.subplots(2, 1, figsize=(3,5), 
    height_ratios=[1.7, 1], sharex=True )
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

ax1, ax2 = axs[0], axs[1]
ax2.pcolor(X, Y, matrix, cmap='plasma')
ax2.set_ylim(1, 16)
pcm = ax1.pcolor(X, Y, matrix, cmap='plasma')
ax1.set_ylim(60, 155)

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax1.set_xscale('log')
ax2.set_xlabel(r'$\omega_0/2\pi$')
ax1.set_ylabel(r'$\omega\tau/2\pi$', loc='bottom')

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

cbar = fig.colorbar(pcm, ax=axs[:])
cbar.set_label(r"$\mathcal{F}$")

cbar.ax.scatter(cbar_marker_shift, ref_fidelity_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_fidelity_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_UA, 'UA', fontsize=8, verticalalignment='center')





# %% plot fidelity (cropped)

cmap = plt.get_cmap('plasma').copy()
cmap.set_extremes(under=OVERMAP_COLOR, over=OVERMAP_COLOR)

upper_limit = ref_fidelity_CD + (ref_fidelity_CD-ref_fidelity_UA)*CD_tolerance
lower_limit = ref_fidelity_UA - (ref_fidelity_CD-ref_fidelity_UA)*UA_tolerance

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'fidelity')
omega_0 = omega_0/(2*np.pi)
X,Y=np.meshgrid(omega_0,osc)

fig, axs = plt.subplots(2, 1, figsize=(3.3,4), 
    height_ratios=[1.2, 1], sharex=True )
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

ax1, ax2 = axs[0], axs[1]
ax2.pcolor(X, Y, matrix, vmin=lower_limit, vmax=upper_limit, cmap=cmap)
ax2.set_ylim(0.5, 16.5)
pcm = ax1.pcolor(X, Y, matrix, vmin=lower_limit, vmax=upper_limit, cmap=cmap)
ax1.set_ylim(60.5, 156)

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax1.set_xscale('log')
ax2.set_xlabel(r'$\omega_0/2\pi$')
ax1.set_ylabel(r'$\omega\tau/2\pi$', loc='bottom')

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

cbar = fig.colorbar(pcm, ax=axs[:], aspect=23, pad=0.1, extend='both', extendfrac=0.04)
cbar.set_label(r"$\mathcal{F}$", rotation=0, labelpad=10)

# add marks
mark_x, mark_y = np.where( matrix > upper_limit )
values = matrix[ np.where( matrix > upper_limit ) ] - upper_limit
if len(values) > 0:
    values = values/np.max(values)
    ax1.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min )
    ax2.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min )
mark_x, mark_y = np.where( matrix < lower_limit )
values = lower_limit - matrix[ np.where( matrix < lower_limit ) ]
if len(values) > 0:
    values = values/np.max(values)
    ax1.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min )
    ax2.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min )

cbar.ax.scatter(cbar_marker_shift, ref_fidelity_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_fidelity_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_fidelity_UA, 'UA', fontsize=8, verticalalignment='center')

cbar.ax.text(0.500, upper_limit+0.003, '+', fontsize=10, c='w', verticalalignment='center', horizontalalignment='center')
cbar.ax.text(0.500, lower_limit-0.002, '-', fontsize=14, c='w', verticalalignment='center', horizontalalignment='center')

plt_titling(fig, suptitle=ax)

if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_fidelity-crop.pdf', bbox_inches='tight')




# %% plot R

cmap = plt.get_cmap('viridis').copy()
cmap.set_extremes(under=OVERMAP_COLOR, over=OVERMAP_COLOR)

upper_limit = ref_R_CD + (ref_R_CD-ref_R_UA)*CD_tolerance
lower_limit = ref_R_UA - (ref_R_CD-ref_R_UA)*UA_tolerance

matrix, osc, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'r')
omega_0 = omega_0/(2*np.pi)
X,Y = np.meshgrid(omega_0,osc)

fig, axs = plt.subplots(2, 1, figsize=(3.3,4), 
    height_ratios=[1.2, 1], sharex=True )
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

ax1, ax2 = axs[0], axs[1]
ax2.pcolor(X, Y, matrix, vmin=lower_limit, vmax=upper_limit, cmap=cmap)
ax2.set_ylim(0.5, 16)
pcm = ax1.pcolor(X, Y, matrix, vmin=lower_limit, vmax=upper_limit, cmap=cmap)
ax1.set_ylim(60, 156)

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax1.set_xscale('log')
ax2.set_xlabel(r'$\omega_0/2\pi$')
ax1.set_ylabel(r'$\omega\tau/2\pi$', loc='bottom')

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

cbar = fig.colorbar(pcm, ax=axs[:], aspect=23, pad=0.1, extend='both', extendfrac=0.04)
cbar.set_label(r"$\mathcal{M}$", rotation=0, labelpad=10)

# add marks
mark_x, mark_y = np.where( matrix > upper_limit )
values = matrix[ np.where( matrix > upper_limit ) ] - upper_limit
if len(values) > 0:
    values = values/np.max(values)
    ax1.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min )
    ax2.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='w', s=excess_marker_size_min )
mark_x, mark_y = np.where( matrix < lower_limit )
values = lower_limit - matrix[ np.where( matrix < lower_limit ) ]
if len(values) > 0:
    values = values/np.max(values)
    ax1.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min )
    ax2.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min )

cbar.ax.scatter(cbar_marker_shift, ref_R_CD, marker='>', c='crimson', linewidth=1, edgecolor='k')
cbar.ax.scatter(cbar_marker_shift, ref_R_UA, marker='o', c='w', linewidth=1, edgecolor='k')
cbar.ax.text(cbar_reflabel_shift, ref_R_CD, 'CD', fontsize=8, verticalalignment='center')
cbar.ax.text(cbar_reflabel_shift, ref_R_UA, 'UA', fontsize=8, verticalalignment='center')

cbar.ax.text(0.500, upper_limit+0.005, '+', fontsize=10, c='w', verticalalignment='center', horizontalalignment='center')
cbar.ax.text(0.500, lower_limit-0.005, '-', fontsize=14, c='w', verticalalignment='center', horizontalalignment='center')

plt_titling(fig, suptitle=ax)

if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_M.pdf', bbox_inches='tight')




# %% plot R - R_CD

def fmt_label(x,p):
    s = f"{100*x:+.0f}"
    if s.endswith("0"):
        s = f"{100*x:.0f}"
    return rf"${s}\%$" if plt.rcParams["text.usetex"] else f"{s} %"

# custom colormap
colr = ['#3b0762', 'w', '#f24500']
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

fig, axs = plt.subplots(2, 1, figsize=(3.3,4), 
    height_ratios=[1.2, 1], sharex=True )
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

ax1, ax2 = axs[0], axs[1]
ax2.pcolor(X, Y, data, vmax=cmap_extr, vmin=-cmap_extr, cmap=cmap)
ax2.set_ylim(0.5, 16+0.5)
pcm = ax1.pcolor(X, Y, data, vmax=cmap_extr, vmin=-cmap_extr, cmap=cmap)
ax1.set_ylim(60-0.5, 156)

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax1.set_xscale('log')
ax2.set_xlabel(r'$\omega_0/2\pi$')
ax1.set_ylabel(r'$\omega\tau/2\pi$', loc='bottom')

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

cbar = fig.colorbar(pcm, ax=axs[:], aspect=25, pad=0.1, extend='both', extendfrac=0.04, format=fmt_label)
cbar.set_label(r"$\mathcal{M}-\mathcal{M}_{\mathrm{CD}}$")

outsider_count = 0
if cmap_extr_limiter is not None:
    upper_limit =  cmap_extr
    lower_limit = -cmap_extr
    mark_x, mark_y = np.where( data > upper_limit )
    values = data[ np.where( data > upper_limit ) ] - upper_limit
    if len(values) > 0:
        outsider_count += len(values)
        values = values/np.max(values)
        ax1.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='k', s=excess_marker_size_min)
        ax2.scatter(omega_0[mark_y], osc[mark_x], marker='+', color='k', s=excess_marker_size_min)
    mark_x, mark_y = np.where( data < lower_limit )
    values = lower_limit - data[ np.where( data < lower_limit ) ]
    if len(values) > 0:
        outsider_count += len(values)
        values = values/np.max(values)
        ax1.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min)
        ax2.scatter(omega_0[mark_y], osc[mark_x], marker='_', color='w', s=excess_marker_size_min)

print('elements inside:', data.shape[0] * data.shape[1] - outsider_count)

if cmap_extr_limiter is not None:
    #cbar.ax.text(2, cmap_extr_limiter*1.1, "$(\pm" + str(cmap_extr_limiter) + ")$", size= 'small' )
    cbar.ax.text(-1, -cmap_extr_limiter*1.3, "$(\mathcal{M}_{\mathrm{CD}} =" + str(np.round(ref_R_CD,2)) + ")$", size= 'small' )
    cbar.ax.text(1.3, upper_limit*1.07, f'$>+{int(cmap_extr_limiter*100)}\%$', fontsize=10, c='k', verticalalignment='center', horizontalalignment='left')
    cbar.ax.text(1.3, lower_limit*1.07, f'$<-{int(cmap_extr_limiter*100)}\%$', fontsize=10, c='k', verticalalignment='center', horizontalalignment='left')

if cmap_extr_limiter is not None:
    cbar.ax.text(0.500, upper_limit+0.0015, '+', fontsize=10, c='w', verticalalignment='center', horizontalalignment='center')
    cbar.ax.text(0.500, lower_limit-0.001, '-', fontsize=14, c='w', verticalalignment='center', horizontalalignment='center')

plt_titling(fig, suptitle=ax)
if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_MvsCD.pdf', bbox_inches='tight')





# %% gain 

def get_contour(matrix, omega_0, omega, threshold):
    Y,X = np.where(matrix >= threshold)
    omega_0 = np.concatenate( (np.array([0.29]), omega_0) ) # WARN: manual correction!!
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

#fig, ax = plt.subplots()
matrix, omega, omega_0 = get_grid(df, key_1 = 'osc', key_2 = 'omega_0', target = 'gain')
omega_0 = omega_0/(2*np.pi)
X,Y = np.meshgrid(omega_0,omega)

scale_value = np.nanmax( np.abs(matrix) )
scale_value = 0.5
retain_value = 0.03
#newcmap = get_cmap_center('RdBu_r', 'gold', retain_value/(scale_value))
newcmap = cm.get_cmap('RdBu_r', 1000)

fig, axs = plt.subplots(2, 1, figsize=(3.3,4), 
    height_ratios=[1.2, 1], sharex=True )
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

ax1, ax2 = axs[0], axs[1]
ax2.pcolor(X, Y, matrix, cmap=newcmap, vmin=-scale_value, vmax=scale_value)
ax2.set_ylim(0.5, 16)
threshold = 0
for polygon in get_contour(matrix, omega_0, omega, threshold):
    ax2.add_patch(mplPolygon(polygon.exterior.coords, 
        fc='none', ec='k', lw=1.7, linestyle='--', clip_on=False))


pcm = ax1.pcolor(X, Y, matrix, cmap=newcmap, vmin=-scale_value, vmax=scale_value)
ax1.set_ylim(60, 156)

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax1.set_xscale('log')
ax2.set_xlabel(r'$\omega_0/2\pi$')
ax1.set_ylabel(r'$\omega\tau/2\pi$', loc='bottom')

d = .3  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

cbar = fig.colorbar(pcm, ax=axs[:], aspect=30, pad=0.1)
cbar.set_label(r"$\mathcal{G}$", rotation=0, labelpad=10)

cbar.ax.text(-1.5,0.23, f"$\ge{int(threshold*100)}\%$", 
    size='x-small', color='k', rotation='vertical', va='center')
rect = patches.Rectangle((-0.2, threshold), 1.15, scale_value-1.5*threshold, 
    linewidth=1.5, edgecolor='k', linestyle='--',
    facecolor='none', clip_on=False, zorder=10)
cbar.ax.add_patch(rect)
plt_titling(fig, suptitle=ax)

if DO_SAVEFIG: plt.savefig(PREFIX + f'oscgr-{MODEL}-{PROTOCOL}-{SCHEDULE}_t{tau}_gain.pdf', bbox_inches='tight')



# %%


