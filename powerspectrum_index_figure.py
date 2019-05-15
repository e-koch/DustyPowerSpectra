
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

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")


# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)


df = pd.read_csv(osjoin(data_path, "pspec_fit_results.csv"), index_col=0)

gals = ['LMC', 'SMC', 'M31', 'M33']

bands = ['', 'MIPS 24', 'MIPS 70', 'PACS 100', 'MIPS 160', 'PACS 160',
         'SPIRE 250', 'SPIRE 350', 'SPIRE 500']

labels = {'orig': "Original", 'mod': "Convolved"}

fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

for i, gal in enumerate(gals):

    ax = axs[i]

    ax.text(0, 2.3, gal, bbox={"boxstyle": "round", "facecolor": "w",
                             "edgecolor": 'k'})

    df_gal = df.loc[[ind for ind in df.index if gal.lower() in ind]]

    # Then split by orig and mod

    for res_type in ['orig', 'mod']:
        df_gal_res = df_gal.loc[[ind for ind in df_gal.index if res_type in ind]]

        df_gal_res.ind.plot(ax=ax, label=labels[res_type],
                            yerr=df_gal_res.ind_std)

    if i == 0:
        ax.legend(frameon=True, loc='lower right')

# Not working in loop?
[ax.grid() for ax in axs]

axs[-1].set_xticklabels(bands)

axs[-1].set_ylim([0.0, 3.0])

fig.text(0.06, 0.5, 'Power-spectrum Index',
         ha='center', va='center', rotation='vertical')

fig.savefig(osjoin(plot_folder, "pspec_index_comparison.png"))
fig.savefig(osjoin(plot_folder, "pspec_index_comparison.pdf"))

plt.close()
