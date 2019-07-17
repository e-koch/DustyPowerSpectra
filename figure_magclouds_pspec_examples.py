
'''
Plot a few example spectra in both galaxies to demonstrate the lack of
break points.
'''

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage as nd
import seaborn as sb
from radio_beam import Beam
import pandas as pd
from glob import glob

osjoin = os.path.join

np.random.seed(43987387)

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
from fit_settings import fitinfo_dict

from plotting_styles import twocolumn_twopanel_figure


twocolumn_twopanel_figure()

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Deal with individual bands first. Column density plots follow.

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

fitinfo_dict = dict()

fitinfo_dict["LMC"] = {'filename': r"lmc_dust.surface.density_FB.beta=1.8_gauss53.4_regrid_mwsub.fits",
                       'beam': Beam(53.4 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 50.1 * u.kpc}
fitinfo_dict["SMC"] = {'filename': r"smc_dust.surface.density_FB.beta=1.8_gauss43.2_regrid_mwsub.fits",
                       'beam': Beam(43.2 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 62.1 * u.kpc}

# y, x pos'ns to use as the LMC and SMC region examples

posns = {'LMC': [(1020, 290), (1020, 800), (153, 494)],
         'SMC': [(153, 693), (255, 591), (357, 591)]}

df = pd.read_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"), index_col=0)

fig, axes = plt.subplots(1, 2, sharex=True)

for i, (gal, ax) in enumerate(zip(fitinfo_dict, axes.ravel())):

    fit_params = df.loc[f"{gal.lower()}"]

    filename = "{0}_coldens.pspec.pkl".format(gal.lower())

    pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))
    pspec.load_beam()

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # One side only shows the power-spectrum
    ax.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10,
              label='Global')
    # And the beam
    phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

    ax.axvline(1 / phys_beam, linestyle=':', linewidth=4,
                  alpha=0.8, color='gray')

    def beam_model(f):
        return gaussian_beam(f, beam_gauss_width)

    beam_amp = 10**(max(fit_params.logA, fit_params.logB) - 2.5)

    # ax.loglog(phys_scales[fit_mask],
    #           beam_amp * beam_model(beam_freqs), 'r:',
    #           linewidth=2)

    # Now plot the example power-spectra and their fits
    df_reg = pd.read_csv(osjoin(data_path, f"{gal}_pspec_perpoint_coldens_fit_results_regionsize_80pix.csv"), index_col=0)

    # for j in range(3):

    #     y = posns[gal][j][0]
    #     x = posns[gal][j][1]

    #     filename_pos = f"{fitinfo_dict[gal]['filename'].rstrip('fits')}_y_{y}_x_{x}_pspec.pkl"

    #     pspec_pos = PowerSpectrum.load_results(osjoin(data_path, gal, 'pspec_coldens_perpoint', filename_pos))
    #     pspec_pos.load_beam()

    #     phys_freqs_pos = pspec_pos._spatial_freq_unit_conversion(pspec_pos.freqs, u.pc**-1).value

    #     phys_scales_pos = 1 / phys_freqs_pos

    #     # One side only shows the power-spectrum
    #     ax.loglog(phys_scales_pos, pspec_pos.ps1D)

    # Or plot all of them in thin lines to get a sense of the distribution
    pspec_files = glob(osjoin(data_path, gal, 'pspec_coldens_perpoint', '*.pkl'))

    rand_order = np.arange(len(pspec_files))
    np.random.shuffle(rand_order)

    for j, pspec_pos_file in enumerate((np.array(pspec_files)[rand_order[:30]])):

        pspec_pos = PowerSpectrum.load_results(pspec_pos_file)
        pspec_pos.load_beam()

        phys_freqs_pos = pspec_pos._spatial_freq_unit_conversion(pspec_pos.freqs, u.pc**-1).value

        phys_scales_pos = 1 / phys_freqs_pos

        # One side only shows the power-spectrum
        if j == 0:
            label = 'Local'
        else:
            label = None
        ax.loglog(phys_scales_pos, pspec_pos.ps1D, 'gray', alpha=1.,
                  linewidth=0.4, label=label)

    ax.loglog(phys_scales,
              beam_amp * beam_model(pspec.freqs.value), 'r:', label='PSF',
              linewidth=2)

    # Invert once b/c all x-axes are the same
    if i == 0:
        ax.invert_xaxis()
        ax.text(0.05, 0.05, f"Dust\nSurface\nDensity",
                size=12,
                bbox={"boxstyle": "round", "facecolor": "w",
                      'edgecolor': 'gray'},
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='left')
    else:
        ax.legend(frameon=True, loc='lower left')

    ax.grid()

    if i == 0:
        ax.set_ylim([1e-4, 3e8])
    else:
        ax.set_ylim([1e-3, 1e7])

    # Galaxy label
    ax.text(0.95, 0.95, gal, size=12,
            bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right')

plt.tight_layout()

fig.text(0.5, 0.04, 'Scale (pc)', ha='center', va='center')
fig.subplots_adjust(left=None, bottom=0.17,
                    right=None, top=None,)

fig.savefig(osjoin(plot_folder, "magcloud_pspec_variation.1Dpspec.png"))
fig.savefig(osjoin(plot_folder, "magcloud_pspec_variation.1Dpspec.pdf"))
plt.close()
