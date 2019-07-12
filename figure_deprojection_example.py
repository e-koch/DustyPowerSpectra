
'''
Appendix figure showing effect of deprojection
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
from scipy.interpolate import InterpolatedUnivariateSpline

osjoin = os.path.join

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

gals = {'M31': 744 * u.kpc, 'M33': 840 * u.kpc}

plot_types = {'orig': ['Original', '-'],
              'dep': ['Deprojected', '--']}

res_type = 'mod'

df = pd.read_csv(osjoin(data_path, "pspec_spire500_deproj_fit_results.csv"), index_col=0)

fig, axes = plt.subplots(1, 2)

for i, (gal, ax) in enumerate(zip(gals, axes.ravel())):

    for ch in ['orig', 'dep']:

        fit_params = df.loc[f"{gal.lower()}_{ch}"]

        if ch == 'orig':
            filename = "{0}_spire500_mod_mjysr.pspec.pkl".format(gal.lower())
        else:
            filename = "{0}_spire500_mod_mjysr_deproj.pspec.pkl".format(gal.lower())

        pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))
        pspec.load_beam()

        beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
        beam_size = beam_size.value
        beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

        high_cut = (1 / (beam_gauss_width * 1.5))
        fit_mask = pspec.freqs.value < high_cut

        # And cut out the largest scales due to expected deviations with
        # small stddev
        fit_mask[:2] = False

        freqs = pspec.freqs.value[fit_mask]
        beam_freqs = pspec.freqs.value[fit_mask]
        ps1D = pspec.ps1D[fit_mask]
        ps1D_stddev = pspec.ps1D_stddev[fit_mask]

        phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

        phys_scales = 1 / phys_freqs

        if ch == 'orig':
            def beam_model(f):
                return gaussian_beam(f, beam_gauss_width)
        else:

            save_name = "{0}_{1}_{2}_mjysr_deproj_beam.pspec.pkl".format(gal.lower(),
                                                                         'spire500',
                                                                         res_type)

            pspec_dep_beam = PowerSpectrum.load_results(osjoin(data_path, gal, save_name))

            spl = InterpolatedUnivariateSpline(pspec_dep_beam.freqs.value,
                                               pspec_dep_beam.ps1D)

            largest_val = pspec_dep_beam.ps1D[0]
            smallest_freq = pspec_dep_beam.freqs.value[0]

            def beam_model(f):

                beam_vals = np.empty_like(f)
                # beam_vals = T.zeros_like(f)
                # if on scales larger than the kernel image, return
                # value on largest scale
                beam_vals[f < smallest_freq] = largest_val
                beam_vals[f >= smallest_freq] = spl(f[f >= smallest_freq])

                return beam_vals


        # beam_amp = 10**(max(fit_params.logA, fit_params.logB) - 1.)

        # Check if the fit uses the PSF
        fit_model = lambda f, args: powerlaw_model(f, *args[:-1]) * beam_model(f)

        ax.loglog(phys_scales, pspec.ps1D, 'k' + plot_types[ch][1],
                  zorder=-10,
                  label=plot_types[ch][0])
        ax.loglog(phys_scales[fit_mask],
                  fit_model(freqs, [fit_params.logA, fit_params.ind,
                                    fit_params.logB]),
                  'r' + plot_types[ch][1],
                  linewidth=3, alpha=0.75)

        # ax.loglog(phys_scales[fit_mask],
        #           beam_amp * beam_model(beam_freqs), 'k:',
        #           linewidth=2)

        if ch == 'orig':
            phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix),
                                                            u.pc**-1).value

            ax.axvline(1 / phys_beam, linestyle=plot_types[ch][1], linewidth=4,
                       alpha=0.8, color='gray')

    ax.grid()
    ax.invert_xaxis()

    if i == 0:
        ax.legend(frameon=True, loc='lower left')

    # Galaxy label
    ax.text(0.9, 0.9, gal, size=12,
            bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
            transform=ax.transAxes,
            verticalalignment='center',
            horizontalalignment='center')

    if i == 1:
        # Add a SPIRE 500 tag

        ax.text(0.175, 0.1, "SPIRE 500", size=12,
                bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
                transform=ax.transAxes,
                verticalalignment='center',
                horizontalalignment='center')

fig.text(0.5, 0.04, 'Scale (pc)', ha='center', va='center')
fig.subplots_adjust(left=None, bottom=0.175,
                    right=None, top=None,)

axes[0].set_ylim([1e-3, 8e10])
axes[1].set_ylim([1e-3, 2e9])

fig.savefig(osjoin(plot_folder, f"deproj_spire500_comparison_1Dpspec.png"))
fig.savefig(osjoin(plot_folder, f"deproj_spire500_comparison_1Dpspec.pdf"))

plt.close()
