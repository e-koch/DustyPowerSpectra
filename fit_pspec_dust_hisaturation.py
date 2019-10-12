
'''
Is HI saturation the source of the difference b/w the dust and HI power
spectra in M33?

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

from turbustat.statistics import PowerSpectrum

from plotting_styles import twocolumn_figure

# Load model functions
repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

twocolumn_figure()

gals = {'LMC': 50.1 * u.kpc, 'SMC': 62.1 * u.kpc,
        'M31': 744 * u.kpc, 'M33': 840 * u.kpc}

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

fitinfo_dict = dict()

fitinfo_dict["LMC"] = {'filename': r"lmc_dust.surface.density_FB.beta=1.8_gauss53.4_regrid_mwsub.fits",
                       'beam': Beam(53.4 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 50.1 * u.kpc,
                       'cosinc': np.cos(34.7 * u.deg),
                       'GDR': 340.}
# fitinfo_dict["SMC"] = {'filename': r"smc_dust.surface.density_FB.beta=1.8_gauss43.2_regrid_mwsub.fits",
#                        'beam': Beam(43.2 * u.arcsec), 'apod_kern': None,
#                        'low_int_cut': None, 'high_int_cut': None,
#                        'low_cut': None, 'high_cut': None,
#                        'distance': 62.1 * u.kpc}
# fitinfo_dict["M31"] = {'filename': r"m31_dust.surface.density_FB.beta=1.8_gauss46.3_regrid_bksub.fits",
#                        'beam': Beam(46.3 * u.arcsec), 'apod_kern': None,
#                        'low_int_cut': None, 'high_int_cut': None,
#                        'low_cut': None, 'high_cut': None,
#                        'distance': 744 * u.kpc}
fitinfo_dict["M33"] = {'filename': r"m33_dust.surface.density_FB.beta=1.8_gauss41.0_regrid_bksub.fits",
                       'beam': Beam(41.0 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 840 * u.kpc,
                       'cosinc': np.cos(55 * u.deg),
                       'GDR': 150.}


for gal in fitinfo_dict:

    filename = osjoin(data_path, gal, fitinfo_dict[gal]['filename'])

    hdu_coldens = fits.open(filename)

    hdr = hdu_coldens[0].header

    # The edges of the maps have high uncertainty. For M31, this may be
    # altering the shape of the power-spectrum. Try removing these edges:
    coldens_mask = np.isfinite(hdu_coldens[0].data[0].squeeze())

    coldens_mask = nd.binary_erosion(coldens_mask,
                                     structure=np.ones((3, 3)),
                                     iterations=8)

    # Get minimal size
    masked_data = hdu_coldens[0].data[0].squeeze()
    masked_data[~coldens_mask] = np.NaN

    proj_coldens = hdu_coldens[0].data[0][nd.find_objects(coldens_mask)[0]]

    hi_satpt = 10.  # Msol / pc^-2
    # hi_satpt = 15. # Msol / pc^-2

    # dense gas to dust conversion of 340 for LMC from Roman-Duval+2014

    gdr = fitinfo_dict[gal]['GDR']

    dust_satpt = (hi_satpt / gdr) / fitinfo_dict[gal]['cosinc'].value
    sat_mask = proj_coldens >= dust_satpt


    dust_sat = proj_coldens.copy()
    dust_sat[sat_mask] = dust_satpt

    pspec = PowerSpectrum(fits.PrimaryHDU(proj_coldens, hdr),
                          distance=fitinfo_dict[gal]['distance'])
    pspec.run(verbose=False, fit_2D=False)

    pspec_sat = PowerSpectrum(fits.PrimaryHDU(dust_sat, hdr),
                              distance=fitinfo_dict[gal]['distance'])
    pspec_sat.run(verbose=False, fit_2D=False)

    beam_size = pspec_sat._beam.major.to(u.deg) / pspec_sat._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width * 3.))

    fit_mask = pspec_sat.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    freqs = pspec_sat.freqs.value[fit_mask]
    ps1D = pspec_sat.ps1D[fit_mask]
    ps1D_stddev = pspec_sat.ps1D_stddev[fit_mask]


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

    tr_plot = pm.traceplot(trace)

    corn_plot = pm.pairplot(trace)

    fig = plt.figure(figsize=(8.4, 2.9))

    phys_freqs = pspec_sat._spatial_freq_unit_conversion(pspec_sat.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # Saturated HI pspec w/fit
    ax = fig.add_subplot(121)

    ax.loglog(phys_scales, pspec_sat.ps1D, 'k', zorder=-10)

    beam_amp = 10**(max(out[0][0],
                        out[0][2]) - 1.)

    ax.loglog(phys_scales[fit_mask],
              fit_model(freqs, *out[0]), 'r--',
              linewidth=3, label='Fit')
    ax.loglog(phys_scales[fit_mask],
              beam_amp * beam_model(freqs), 'r:', label='PSF')

    ax.legend(frameon=True, loc='upper right')

    ax.invert_xaxis()

    # Orig vs. saturated pspec.
    ax2 = fig.add_subplot(122)

    ax2.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10, label='Original')
    ax2.loglog(phys_scales, pspec_sat.ps1D, 'k--', zorder=-10,
               label='Saturated')

    beam_amp = 10**(max(out[0][0],
                        out[0][2]) - 1.)

    ax2.loglog(phys_scales[fit_mask],
               beam_amp * beam_model(freqs), 'r:', label='PSF')

    ax2.legend(frameon=True, loc='upper right')

    ax2.invert_xaxis()

    fig_im = plt.figure(figsize=(8.4, 5.0))

    ax = fig_im.add_subplot(121)

    im = ax.imshow(proj_coldens, origin='lower')
    ax.contour(sat_mask, colors='c')
    plt.colorbar(im, ax=ax)

    ax2 = fig_im.add_subplot(122)

    im2 = ax2.imshow(dust_sat, origin='lower')
    plt.colorbar(im2, ax=ax2)

    input("?")
