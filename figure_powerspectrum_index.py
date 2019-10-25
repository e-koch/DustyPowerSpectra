
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

# Broken plaw fit params for MIPS 24
df_bplaw = pd.read_csv(osjoin(data_path, 'model_comparison',
                              "pspec_modecompare_fit_results.csv"),
                       index_col=0)


gals = ['LMC', 'SMC', 'M31', 'M33']

bands = ['MIPS 24', 'MIPS 70', 'PACS 100', 'MIPS 160', 'PACS 160',
         'SPIRE 250', 'SPIRE 350', 'SPIRE 500', r'$\Sigma_{\rm dust}$']

labels = {'orig': "Original", 'mod': "Convolved"}

fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

col_pal = sb.color_palette()

for i, gal in enumerate(gals):

    ax = axs[i]

    ax.text(-0.3, 2.3, gal, bbox={"boxstyle": "round", "facecolor": "w",
                                  "edgecolor": 'k'})

    df_gal = df.loc[[ind for ind in df.index if gal.lower() in ind]]
    df_gal.index = [ind.replace("_", " ") for ind in df_gal.index]

    df_gal_brok = df_bplaw.loc[[ind for ind in df_bplaw.index
                                if gal.lower() in ind]]

    # Then split by orig and mod

    for j, res_type in enumerate(['orig', 'mod']):
        df_gal_res = df_gal.loc[[ind for ind in df_gal.index
                                 if res_type in ind]]

        df_gal_brok_res = df_gal_brok.loc[[ind for ind in df_gal_brok.index
                                           if res_type in ind]]

        # Plot both plaws for LMC MIPS 24 um
        if gal == 'LMC':
            print(df_gal_brok_res.iloc[2*j].index1_brokplaw)
            print(df_gal_brok_res.iloc[2*j].index2_brokplaw)
            print(df_gal_brok_res.iloc[2*j+1].index1_brokplaw)
            print(df_gal_brok_res.iloc[2*j+1].index2_brokplaw)

            x_pos = -0.2 if j == 0 else 0.2

            # MIPS 24um is up first
            ax.errorbar([x_pos], df_gal_brok_res.iloc[2 * j].index1_brokplaw,
                        yerr=df_gal_brok_res.iloc[2 * j + 1].index1_brokplaw,
                        color=col_pal[j], marker='s')

            ax.errorbar([x_pos], df_gal_brok_res.iloc[2 * j].index2_brokplaw,
                        yerr=df_gal_brok_res.iloc[2 * j + 1].index2_brokplaw,
                        color=col_pal[j], marker='D')

            ax.errorbar(np.arange(1, 8), df_gal_res.ind[1:],
                        yerr=df_gal_res.ind_std[1:],
                        linestyle='-' if res_type == 'orig' else '--',
                        label=labels[res_type],
                        color=col_pal[j])

        else:

            ax.errorbar(np.arange(8), df_gal_res.ind, yerr=df_gal_res.ind_std,
                        linestyle='-' if res_type == 'orig' else '--',
                        label=labels[res_type],
                        color=col_pal[j])

        # df_gal_res.ind.plot(ax=ax, label=labels[res_type],
        #                     yerr=df_gal_res.ind_std,
        #                     linestyle='--',
        #                     rot=45)

    if i == 0:
        ax.legend(frameon=True, loc='lower right')

    # And the dust SD
    ax.errorbar(8., df_coldens.ind[gal.lower()],
                yerr=df_coldens.ind_std[gal.lower()],
                marker='o',
                color=col_pal[2])

# Not working in loop?
[ax.grid() for ax in axs]

axs[-1].set_xticks(np.arange(9))
axs[-1].set_xticklabels(bands)
plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45,
         horizontalalignment='right')

axs[-1].set_ylim([0.0, 3.0])
axs[-1].set_xlim([-0.6, 8.6])

fig.text(0.03, 0.5, r'Power Spectrum Index ($\beta$)',
         ha='center', va='center', rotation='vertical')

plt.subplots_adjust(bottom=0.13)

fig.savefig(osjoin(plot_folder, "pspec_index_comparison.png"))
fig.savefig(osjoin(plot_folder, "pspec_index_comparison.pdf"))

plt.close()
