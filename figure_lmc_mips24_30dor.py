
'''
MIPS 24 convolved showing LMC, w/o and w/ 30 Dor
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import astropy.units as u
import os
from os.path import join as osjoin
import pandas as pd
from turbustat.statistics import PowerSpectrum

from plotting_styles import onecolumn_figure


repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

gals = {'LMC': 50.1 * u.kpc}

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

gal = 'LMC'

lmc_mips24_slice = {"30dor": ["With 30 Dor", '--'],
                    'no30dor': ["Without 30 Dor", ':']}

name = 'mips24'

res_type = 'mod'

onecolumn_figure()

fig = plt.figure()

ax = fig.add_subplot(111)

filename = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(),
                                                name,
                                                res_type)

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

# One side only shows the power-spectrum
ax.loglog(phys_scales, pspec.ps1D / pspec.ps1D.max(),
          'k', zorder=-10, label='LMC')

# W/ 30 Dor

for i, slice_name in enumerate(lmc_mips24_slice):

    filename = "{0}_{1}_{2}_{3}_mjysr.pspec.pkl".format(gal.lower(),
                                                        name,
                                                        res_type,
                                                        slice_name)

    pspec_reg = PowerSpectrum.load_results(osjoin(data_path, gal, filename))
    pspec_reg.load_beam()

    beam_size_reg = pspec_reg._beam.major.to(u.deg) / pspec_reg._ang_size.to(u.deg)
    beam_size_reg = beam_size_reg.value
    beam_gauss_width_reg = beam_size_reg / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width_reg * 3.))
    fit_mask_reg = pspec_reg.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask_reg[:2] = False

    freqs = pspec_reg.freqs.value[fit_mask_reg]
    ps1D = pspec_reg.ps1D[fit_mask_reg]
    ps1D_stddev = pspec_reg.ps1D_stddev[fit_mask_reg]

    phys_freqs_reg = pspec_reg._spatial_freq_unit_conversion(pspec_reg.freqs, u.pc**-1).value

    phys_scales_reg = 1 / phys_freqs_reg

    ax.loglog(phys_scales_reg, pspec_reg.ps1D / pspec_reg.ps1D.max() / 10**(2*i+2),
              'k' + lmc_mips24_slice[slice_name][1],
              zorder=-10,
              label=lmc_mips24_slice[slice_name][0])

# And the beam
phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

ax.axvline(1 / phys_beam, linestyle=':', linewidth=4,
           alpha=0.8, color='gray')

def beam_model(f):
    return gaussian_beam(f, beam_gauss_width)

ax.loglog(phys_scales[fit_mask],
          1e-8 * beam_model(beam_freqs), 'r-.',
          linewidth=2,
          label='PSF')

# Invert once b/c all x-axes are the same
ax.invert_xaxis()
ax.grid()

# Galaxy label
ax.text(0.95, 0.95, "MIPS 24", size=12,
        bbox={"boxstyle": "round", "facecolor": "w", 'edgecolor': 'gray'},
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right')

ax.set_xlabel("Scale (pc)")
ax.set_ylabel("Normalized Power")

ax.legend(frameon=True, ncol=2, loc='lower center')

ax.set_ylim([5e-13, 2])

plt.tight_layout()

fig.savefig(osjoin(plot_folder, f"lmc_mips24_wo_30dor_1Dpspec.png"))
fig.savefig(osjoin(plot_folder, f"lmc_mips24_wo_30dor_1Dpspec.pdf"))

plt.close()
