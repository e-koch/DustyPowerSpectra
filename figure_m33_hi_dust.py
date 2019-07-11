
'''
Make a figure comparing the dust and HI power-spectra in M33
to demonstrate how different they are.
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

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
from fit_settings import fitinfo_dict

from plotting_styles import onecolumn_figure


onecolumn_figure()

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Deal with individual bands first. Column density plots follow.

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

gals = {'M33': 840 * u.kpc}

df = pd.read_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"), index_col=0)
df_hi = pd.read_csv(osjoin(data_path, "pspec_hi_conv_m33_fit_results.csv"), index_col=0)

# Open the pspec files
hi_name = osjoin(data_path,
                 "M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.moment0_Kkms.fits")

hi_pspec_name = f"{hi_name.rstrip('fits')}conv.pspec.pkl"

hi_pspec = PowerSpectrum.load_results(hi_pspec_name)
hi_pspec.load_beam()

dust_pspec_name = osjoin(data_path, 'M33', "m33_coldens.pspec.pkl")

dust_pspec = PowerSpectrum.load_results(dust_pspec_name)
dust_pspec.load_beam()

dust_norm = dust_pspec.ps1D.max()
hi_norm = hi_pspec.ps1D.max()

fig = plt.figure()

ax = fig.add_subplot(111)

gal = 'M33'

fit_params = df.loc[f"{gal.lower()}"]

beam_size = dust_pspec._beam.major.to(u.deg) / dust_pspec._ang_size.to(u.deg)
beam_size = beam_size.value
beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

high_cut = (1 / (beam_gauss_width * 1.5))
fit_mask = dust_pspec.freqs.value < high_cut

# And cut out the largest scales due to expected deviations with
# small stddev
fit_mask[:2] = False

freqs = dust_pspec.freqs.value[fit_mask]
beam_freqs = dust_pspec.freqs.value[fit_mask]
ps1D = dust_pspec.ps1D[fit_mask]
ps1D_stddev = dust_pspec.ps1D_stddev[fit_mask]

phys_freqs = dust_pspec._spatial_freq_unit_conversion(dust_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, dust_pspec.ps1D / dust_norm, 'k', zorder=-10, label='Dust')
# And the beam
phys_beam = dust_pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

ax.axvline(1 / phys_beam, linestyle='-', linewidth=4,
           alpha=0.6, color='gray')

def beam_model(f):
    return gaussian_beam(f, beam_gauss_width)

fit_model = lambda f, args: powerlaw_model(f, *args[:-1]) * beam_model(f) + 10**args[-1]

# ax.loglog(phys_scales[fit_mask],
#           fit_model(freqs, [fit_params.logA, fit_params.ind,
#                             fit_params.logB, fit_params.logC]) / dust_norm,
#           'r--',
#           linewidth=3)

# HI

fit_params_hi = df_hi.loc[f"{gal.lower()}"]

hi_beam_size = hi_pspec._beam.major.to(u.deg) / hi_pspec._ang_size.to(u.deg)
hi_beam_size = hi_beam_size.value
hi_beam_gauss_width = hi_beam_size / np.sqrt(8 * np.log(2))

high_cut = (1 / (hi_beam_gauss_width * 3))
fit_mask = hi_pspec.freqs.value < high_cut

# And cut out the largest scales due to expected deviations with
# small stddev
fit_mask[:2] = False

freqs = hi_pspec.freqs.value[fit_mask]
beam_freqs = hi_pspec.freqs.value[fit_mask]
ps1D = hi_pspec.ps1D[fit_mask]
ps1D_stddev = hi_pspec.ps1D_stddev[fit_mask]

phys_freqs = hi_pspec._spatial_freq_unit_conversion(hi_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, hi_pspec.ps1D / hi_norm, 'k--', zorder=-10, label=r'{\sc HI}')
# And the beam
# phys_beam = hi_pspec._spatial_freq_unit_conversion(1 / (hi_beam_size * u.pix), u.pc**-1).value

# ax.axvline(1 / phys_beam, linestyle=':', linewidth=4,
#            alpha=0.6, color='gray')

def beam_model(f):
    return gaussian_beam(f, hi_beam_gauss_width)

# fit_model_hi = lambda f, args: powerlaw_model(f, *args[:-1]) * beam_model(f)

# ax.loglog(phys_scales[fit_mask],
#           fit_model_hi(freqs, [fit_params_hi.logA, fit_params_hi.ind,
#                                fit_params_hi.logB]) / hi_norm,
#           'r--',
#           linewidth=3)


ax.loglog(phys_scales[fit_mask],
          1e-7 * beam_model(freqs), 'r:', label='PSF',
          linewidth=2)


ax.text(0.95, 0.95, gal, size=12,
        bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right')

# Invert once b/c all x-axes are the same
ax.invert_xaxis()
ax.grid()

ax.legend(frameon=True)

plt.xlabel("Scale (pc)")
plt.ylabel("Normalized Power")

plt.tight_layout()

plt.xlim([2e4, 100])

fig.savefig(osjoin(plot_folder, f"m33_hi_dust_1Dpspec.png"))
fig.savefig(osjoin(plot_folder, f"m33_hi_dust_1Dpspec.pdf"))

plt.close()
