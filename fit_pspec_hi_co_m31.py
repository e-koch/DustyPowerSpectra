
'''
Make total gas surface density maps for M31 and fit their power spectra.
Also do the CO(1-0) map.
'''


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from radio_beam import Beam
import seaborn as sb
import astropy.units as u
import os
from os.path import join as osjoin
import scipy.ndimage as nd
import pymc3 as pm
from spectral_cube import Projection

from turbustat.statistics import PowerSpectrum

from plotting_styles import twocolumn_figure

# Load model functions
repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# For M33, load in the HI and CO and make a gas surface density map with
# default-ish conversion factors.

data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

hi_name = osjoin(data_path, "M31_HI",
                 "M31_14A_HI_contsub_width_04kms.image.pbcor.mom0.Kkms.fits")

co_name = osjoin(data_path, "M31_CO",
                 "m31_iram_Kkms.fits")

co10_mass_conversion = 4.8 * (u.Msun / u.pc ** 2) / (u.K * u.km / u.s)

# Note that the top two conversions contain a 1.4x correction for He.
# So they will give the atomic mass, not the HI mass!
hi_mass_conversion = 0.0196 * (u.M_sun / u.pc ** 2) / (u.K * u.km / u.s)

hi_proj = Projection.from_hdu(fits.open(hi_name))

# Convolve co_proj to the HI beam
co_proj = Projection.from_hdu(fits.open(co_name)).to(u.K * u.km / u.s)
co_proj[np.isnan(co_proj)] = 0.
co_proj[co_proj.value < 0.] = 0.
co_proj = co_proj.convolve_to(hi_proj.beam)

co_proj = co_proj.reproject(hi_proj.header)

do_makepspec = False
do_fitpspec = False
do_makepspec_doub_aco = False
do_fitpspec_doub_aco = False
do_makepspec_co = True
do_fitpspec_co = True

if do_makepspec:

    pspec_name = osjoin(data_path, 'M31_CO', 'm33_hi_co_dustSD.pspec.pkl')

    gas_sd = hi_proj * hi_mass_conversion + co10_mass_conversion * co_proj
    gas_sd[np.isnan(hi_proj)] = np.NaN

    pspec = PowerSpectrum(gas_sd, distance=720 * u.kpc)

    pspec.run(verbose=False, fit_2D=False, high_cut=10**-1.3 / u.pix)
    # pspec.plot_fit()

    pspec.save_results(pspec_name)

# Fit the pspec.
if do_fitpspec:

    pspec = PowerSpectrum.load_results(pspec_name)
    pspec.load_beam()

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width * 3.))

    fit_mask = pspec.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    freqs = pspec.freqs.value[fit_mask]
    ps1D = pspec.ps1D[fit_mask]
    ps1D_stddev = pspec.ps1D_stddev[fit_mask]

    def beam_model(f):
        return gaussian_beam(f, beam_gauss_width)

    nsamp = 6000

    fixB = False

    noise_term = False

    out, summ, trace, fit_model = \
        fit_pspec_model(freqs, ps1D,
                        ps1D_stddev,
                        beam_model=beam_model,
                        nsamp=nsamp,
                        fixB=fixB,
                        noise_term=noise_term)

    print(out)
    # w/ a_CO = 4.8
    # [array([ 5.58047543,  2.3407672 , -6.76021778]),
    # array([0.14475136, 0.10946982, 7.77337036])]

    tr_plot = pm.traceplot(trace)

    corn_plot = pm.pairplot(trace)

    fig = plt.figure(figsize=(4.2, 2.9))

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # Saturated HI pspec w/fit
    ax = fig.add_subplot(111)

    ax.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

    beam_amp = 10**(max(out[0][0],
                        out[0][2]) - 1.)

    ax.loglog(phys_scales[fit_mask],
              fit_model(freqs, *out[0]), 'r--',
              linewidth=3, label='Fit')
    ax.loglog(phys_scales[fit_mask],
              beam_amp * beam_model(freqs), 'r:', label='PSF')

    ax.legend(frameon=True, loc='upper right')

    ax.invert_xaxis()

if do_makepspec_doub_aco:

    pspec_name = osjoin(data_path, 'M31_CO', 'm33_hi_co_dustSD.pspec_doub_aco.pkl')

    gas_sd = hi_proj * hi_mass_conversion + 2 * co10_mass_conversion * co_proj
    gas_sd[np.isnan(hi_proj)] = np.NaN

    pspec = PowerSpectrum(gas_sd, distance=720 * u.kpc)

    pspec.run(verbose=False, fit_2D=False, high_cut=10**-1.3 / u.pix)
    # pspec.plot_fit()

    pspec.save_results(pspec_name)

if do_fitpspec_doub_aco:

    pspec_name = osjoin(data_path, 'M31_CO', 'm33_hi_co_dustSD.pspec_doub_aco.pkl')

    pspec = PowerSpectrum.load_results(pspec_name)
    pspec.load_beam()

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width * 3.))

    fit_mask = pspec.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    freqs = pspec.freqs.value[fit_mask]
    ps1D = pspec.ps1D[fit_mask]
    ps1D_stddev = pspec.ps1D_stddev[fit_mask]

    def beam_model(f):
        return gaussian_beam(f, beam_gauss_width)

    nsamp = 6000

    fixB = False

    noise_term = False

    out, summ, trace, fit_model = \
        fit_pspec_model(freqs, ps1D,
                        ps1D_stddev,
                        beam_model=beam_model,
                        nsamp=nsamp,
                        fixB=fixB,
                        noise_term=noise_term)

    print(out)
    # w/ a_CO = 9.6
    # [array([ 6.20749931,  2.11465308, -4.43332322]),
    # array([0.12928604, 0.09980371, 7.56877381])]



    tr_plot = pm.traceplot(trace)

    corn_plot = pm.pairplot(trace)

    fig = plt.figure(figsize=(4.2, 2.9))

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # Saturated HI pspec w/fit
    ax = fig.add_subplot(111)

    ax.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

    beam_amp = 10**(max(out[0][0],
                        out[0][2]) - 1.)

    ax.loglog(phys_scales[fit_mask],
              fit_model(freqs, *out[0]), 'r--',
              linewidth=3, label='Fit')
    ax.loglog(phys_scales[fit_mask],
              beam_amp * beam_model(freqs), 'r:', label='PSF')

    ax.legend(frameon=True, loc='upper right')

    ax.invert_xaxis()

if do_makepspec_co:

    pspec_name = osjoin(data_path, 'M31_CO', 'm31_co.pspec.pkl')

    # And with CO
    pspec = PowerSpectrum(co_proj,
                          distance=720 * u.kpc)

    pspec.run(verbose=False, fit_2D=False, high_cut=10**-1.3 / u.pix)
    # pspec.plot_fit()

    pspec.save_results(pspec_name)

if do_fitpspec_co:

    pspec_name = osjoin(data_path, 'M31_CO', 'm31_co.pspec.pkl')

    pspec = PowerSpectrum.load_results(pspec_name)
    pspec.load_beam()

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width * 3.))

    fit_mask = pspec.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    freqs = pspec.freqs.value[fit_mask]
    ps1D = pspec.ps1D[fit_mask]
    ps1D_stddev = pspec.ps1D_stddev[fit_mask]

    def beam_model(f):
        return gaussian_beam(f, beam_gauss_width)

    nsamp = 6000

    fixB = False

    noise_term = False

    out, summ, trace, fit_model = \
        fit_pspec_model(freqs, ps1D,
                        ps1D_stddev,
                        beam_model=beam_model,
                        nsamp=nsamp,
                        fixB=fixB,
                        noise_term=noise_term)

    print(out)
    # [array([ 4.6017837 ,  1.58634644, -7.41798926]),
    # array([0.10044863, 0.07508173, 7.19343479])]


    tr_plot = pm.traceplot(trace)

    corn_plot = pm.pairplot(trace)

    fig = plt.figure(figsize=(4.2, 2.9))

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # Saturated HI pspec w/fit
    ax = fig.add_subplot(111)

    ax.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

    beam_amp = 10**(max(out[0][0],
                        out[0][2]) - 1.)

    ax.loglog(phys_scales[fit_mask],
              fit_model(freqs, *out[0]), 'r--',
              linewidth=3, label='Fit')
    ax.loglog(phys_scales[fit_mask],
              beam_amp * beam_model(freqs), 'r:', label='PSF')

    ax.legend(frameon=True, loc='upper right')

    ax.invert_xaxis()
