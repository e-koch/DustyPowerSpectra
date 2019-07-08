
'''
Plot index vs. band for each galaxy in one plot
'''

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage as nd
import seaborn as sb
from radio_beam import Beam
import pymc3 as pm
import pandas as pd
from spectral_cube import Projection

osjoin = os.path.join

from plotting_styles import onecolumn_Npanel_figure


onecolumn_Npanel_figure(N=2)

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")


# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)


df = pd.read_csv(osjoin(data_path, "pspec_fit_results.csv"), index_col=0)
df_coldens = pd.read_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"),
                         index_col=0)

gals = ['LMC', 'SMC', 'M31', 'M33']

bands = ['MIPS 24', 'MIPS 70', 'PACS 100', 'MIPS 160', 'PACS 160',
         'SPIRE 250', 'SPIRE 350', 'SPIRE 500', r'$\Sigma_{\rm dust}$']

labels = {'orig': "Original", 'mod': "Convolved"}

fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

for i, gal in enumerate(gals):

    ax = axs[i]

    ax.text(-0.3, 2.3, gal, bbox={"boxstyle": "round", "facecolor": "w",
                               "edgecolor": 'k'})

    df_gal = df.loc[[ind for ind in df.index if gal.lower() in ind]]
    df_gal.index = [ind.replace("_", " ") for ind in df_gal.index]

    # Then split by orig and mod

    for res_type in ['orig', 'mod']:
        df_gal_res = df_gal.loc[[ind for ind in df_gal.index if res_type in ind]]

        ax.errorbar(np.arange(8), df_gal_res.ind, yerr=df_gal_res.ind_std,
                    linestyle='-' if res_type == 'orig' else '--',
                    label=labels[res_type])

        # df_gal_res.ind.plot(ax=ax, label=labels[res_type],
        #                     yerr=df_gal_res.ind_std,
        #                     linestyle='--',
        #                     rot=45)

    if i == 0:
        ax.legend(frameon=True, loc='lower right')

    # And the dust SD
    ax.errorbar(8., df_coldens.ind[gal.lower()],
                yerr=df_coldens.ind_std[gal.lower()],
                marker='o')

# Not working in loop?
[ax.grid() for ax in axs]

axs[-1].set_xticks(np.arange(9))
axs[-1].set_xticklabels(bands)
plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45,
         horizontalalignment='right')

axs[-1].set_ylim([0.0, 3.0])
axs[-1].set_xlim([-0.6, 8.6])

fig.text(0.03, 0.5, r'Power-spectrum Index ($\beta$)',
         ha='center', va='center', rotation='vertical')

plt.subplots_adjust(bottom=0.13)

fig.savefig(osjoin(plot_folder, "pspec_index_comparison.png"))
fig.savefig(osjoin(plot_folder, "pspec_index_comparison.pdf"))

plt.close()
