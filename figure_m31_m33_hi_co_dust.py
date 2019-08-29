
'''
Make a figure comparing the HI, CO and dust power spectra

Two panels. One for M31 (top), second for M33 (bottom).
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


repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
from fit_settings import fitinfo_dict

from plotting_styles import onecolumn_twopanel_figure


onecolumn_twopanel_figure(font_scale=1.2)

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Deal with individual bands first. Column density plots follow.

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)


fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# M31

# Open the pspec files
hi_pspec_name = osjoin(data_path, "M31_HI",
                 "M31_14A_HI_contsub_width_04kms.image.pbcor.mom0.Kkms.pspec.pkl")

hi_pspec = PowerSpectrum.load_results(hi_pspec_name)
hi_pspec.load_beam()

dust_pspec_name = osjoin(data_path, 'M31', 'm31_dust_hi_match.pspec.pkl')

dust_pspec = PowerSpectrum.load_results(dust_pspec_name)
dust_pspec.load_beam()

co_pspec_name = osjoin(data_path, 'M31_CO', 'm31_co.pspec.pkl')

co_pspec = PowerSpectrum.load_results(co_pspec_name)
co_pspec.load_beam()

dust_norm = dust_pspec.ps1D.max()
hi_norm = hi_pspec.ps1D.max()
co_norm = co_pspec.ps1D.max()

ax = axs[0]

gal = 'M31'

# CO pspec
phys_freqs = co_pspec._spatial_freq_unit_conversion(co_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, co_pspec.ps1D / co_norm, 'k:', zorder=-10, label='CO(1-0)')

# dust
beam_size = dust_pspec._beam.major.to(u.deg) / dust_pspec._ang_size.to(u.deg)
beam_size = beam_size.value
beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

phys_freqs = dust_pspec._spatial_freq_unit_conversion(dust_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, dust_pspec.ps1D / dust_norm, 'k', zorder=-10, label='Dust')
# And the beam
phys_beam = dust_pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

ax.axvline(1 / phys_beam, linestyle='-', linewidth=4,
           alpha=0.6, color='gray')

# Add a label
ax.text(1.1 / phys_beam, 1e-3, f"{int(round(1 / phys_beam))} pc", color='gray',
        rotation=90, horizontalalignment='right',
        size=13)

# And HI plus the PSF

def beam_model(f):
    return gaussian_beam(f, beam_gauss_width)

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

def beam_model(f):
    return gaussian_beam(f, hi_beam_gauss_width)

ax.loglog(phys_scales[fit_mask],
          1e-6 * beam_model(freqs), 'r:', label='PSF',
          linewidth=2)

ax.text(0.95, 0.95, gal, size=13,
        bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right')

# Invert once b/c all x-axes are the same
# ax.invert_xaxis()
ax.grid()

ax.legend(frameon=True, loc='center left')


# M33

# Open the pspec files
hi_name = osjoin(data_path, "M33_HI",
                 "M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.moment0_Kkms.fits")

hi_pspec_name = f"{hi_name.rstrip('fits')}conv.pspec.pkl"

hi_pspec = PowerSpectrum.load_results(hi_pspec_name)
hi_pspec.load_beam()

dust_pspec_name = osjoin(data_path, 'M33', "m33_coldens.pspec.pkl")

dust_pspec = PowerSpectrum.load_results(dust_pspec_name)
dust_pspec.load_beam()

co_pspec_name = osjoin(data_path, 'M33_CO', 'm33_co.pspec.pkl')

co_pspec = PowerSpectrum.load_results(co_pspec_name)
co_pspec.load_beam()


dust_norm = dust_pspec.ps1D.max()
hi_norm = hi_pspec.ps1D.max()
co_norm = co_pspec.ps1D.max()

ax = axs[1]

gal = 'M33'

# CO pspec
phys_freqs = co_pspec._spatial_freq_unit_conversion(co_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, co_pspec.ps1D / co_norm, 'k:', zorder=-10, label='CO(2-1)')

# dust
beam_size = dust_pspec._beam.major.to(u.deg) / dust_pspec._ang_size.to(u.deg)
beam_size = beam_size.value
beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

phys_freqs = dust_pspec._spatial_freq_unit_conversion(dust_pspec.freqs, u.pc**-1).value

phys_scales = 1 / phys_freqs

# One side only shows the power-spectrum
ax.loglog(phys_scales, dust_pspec.ps1D / dust_norm, 'k', zorder=-10, label='Dust')
# And the beam
phys_beam = dust_pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

ax.axvline(1 / phys_beam, linestyle='-', linewidth=4,
           alpha=0.6, color='gray')
# Add a label
ax.text(1.1 / phys_beam, 1e-3, f"{int(round(1 / phys_beam))} pc", color='gray',
        rotation=90, horizontalalignment='right',
        size=13)

# And HI plus the PSF

def beam_model(f):
    return gaussian_beam(f, beam_gauss_width)

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

def beam_model(f):
    return gaussian_beam(f, hi_beam_gauss_width)

ax.loglog(phys_scales[fit_mask],
          1e-6 * beam_model(freqs), 'r:', label='PSF',
          linewidth=2)

ax.text(0.95, 0.95, gal, size=13,
        bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right')

# Invert once b/c all x-axes are the same
# ax.invert_xaxis()
ax.grid()

ax.set_ylim([1e-8, 2])
ax.set_xlim([5e4, 100])

ax.legend(frameon=True, loc='center left')

ax.set_xlabel("Scale (pc)")
fig.text(0.04, 0.53, 'Normalized Power', ha='center', va='center', rotation=90)

axs[0].invert_xaxis()
axs[1].invert_xaxis()

fig.subplots_adjust(left=0.16, bottom=0.08,
                    right=0.98, top=0.98,
                    wspace=0.01, hspace=0.05)

# plt.tight_layout()

fig.savefig(osjoin(plot_folder, f"m31_m33_hi_co_dust_1Dpspec.png"))
fig.savefig(osjoin(plot_folder, f"m31_m33_hi_co_dust_1Dpspec.pdf"))

plt.close()
