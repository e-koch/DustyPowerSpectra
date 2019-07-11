
'''
Deproject the M33 and M31 column density maps and compare to the originals.
'''

import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from spectral_cube import Projection
from radio_beam import Beam
import scipy.ndimage as nd
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import InterpolatedUnivariateSpline
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from galaxies import Galaxy
from cube_analysis.cube_deproject import deproject
from turbustat.statistics import PowerSpectrum

# Load model functions
repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

osjoin = os.path.join

data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
# data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

fitinfo_dict = dict()

fitinfo_dict["M33"] = {'filename': r"m33_dust.surface.density_FB.beta=1.8_gauss41.0_regrid_bksub.fits",
                       'beam': Beam(41.0 * u.arcsec), 'apod_kern': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 840 * u.kpc}
fitinfo_dict["M31"] = {'filename': r"m31_dust.surface.density_FB.beta=1.8_gauss46.3_regrid_bksub.fits",
                       'beam': Beam(46.3 * u.arcsec), 'apod_kern': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 744 * u.kpc}

ncores = 1

nsamp = 2000
# nsamp = 6000

row_names = []

fit_results = {'logA': [], 'ind': [], 'logB': [], 'logC': [],
               'logA_std': [], 'ind_std': [], 'logB_std': [],
               'logC_std': []}

# for gal in ['M33', 'M31']:
for gal in ['M31']:

    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    gal_obj = Galaxy(gal)
    gal_obj.distance = fitinfo_dict[gal]['distance']

    if gal == 'M31':
        # Add 180 deg to the PA
        gal_obj.position_angle += 180 * u.deg


    filename = osjoin(data_path, gal, fitinfo_dict[gal]['filename'])

    hdu_coldens = fits.open(filename)

    proj_coldens = Projection.from_hdu(fits.PrimaryHDU(hdu_coldens[0].data[0].squeeze(),
                                                       hdu_coldens[0].header))

    # Get minimal size
    proj_coldens = proj_coldens[nd.find_objects(np.isfinite(proj_coldens))[0]]

    proj_coldens = proj_coldens.with_beam(fitinfo_dict[gal]['beam'])

    # Run on the original image

    pspec = PowerSpectrum(proj_coldens,
                          distance=fitinfo_dict[gal]['distance'])
    pspec.run(verbose=False, beam_correct=False, fit_2D=False,
              high_cut=0.1 / u.pix,
              use_pyfftw=True, threads=ncores,
              # radial_pspec_kwargs={"theta_0": gal_obj.position_angle + 90*u.deg, "delta_theta": 50 * u.deg},
              # radial_pspec_kwargs={"logspacing": True, 'binsize': 10.},
              apodize_kernel=fitinfo_dict[gal]['apod_kern'])

    # Deproject the image AND the beam

    deproj_img = deproject(proj_coldens, proj_coldens.header,
                           gal_obj, inc_correction=True)

    # Cut this image down to the minimal size
    deproj_img = deproj_img[nd.find_objects(np.isfinite(deproj_img))[0]]

    # Just keep the original header. Doesn't matter until the
    # conversion to phys units
    deproj_img_hdu = fits.PrimaryHDU(deproj_img, proj_coldens.header)

    # The beam is symmetric so we only need to warp it to match the PSF shape
    # in the deprojected frame
    # That's just an elliptical beam, so define a new beam object
    inc = gal_obj.inclination
    deproj_beam = Beam(major=proj_coldens.beam.major / np.cos(inc).value,
                       minor=proj_coldens.beam.major,
                       pa=90 * u.deg)

    pixscale = np.abs(proj_plane_pixel_scales(proj_coldens.wcs.celestial)[0]) * u.deg
    beam_arr = deproj_beam.as_kernel(pixscale, x_size=deproj_img.shape[1],
                                     y_size=deproj_img.shape[0])

    # Make an HDU for the beam so we can make a model of its power-spectrum
    deproj_beam_hdu = fits.PrimaryHDU(beam_arr)

    pspec_dep_beam = PowerSpectrum(deproj_beam_hdu)
    pspec_dep_beam.run(verbose=False, beam_correct=False, fit_2D=False,
                       high_cut=0.1 / u.pix,
                       use_pyfftw=True, threads=ncores)

    # Make an interpolated model from the ps1D of the beam
    spl = InterpolatedUnivariateSpline(pspec_dep_beam.freqs.value,
                                       pspec_dep_beam.ps1D)

    largest_val = pspec_dep_beam.ps1D[0]
    smallest_freq = pspec_dep_beam.freqs.value[0]

    def beam_model_dep(f):

        beam_vals = np.empty_like(f)
        # beam_vals = T.zeros_like(f)
        # if on scales larger than the kernel image, return
        # value on largest scale
        beam_vals[f < smallest_freq] = largest_val
        beam_vals[f >= smallest_freq] = spl(f[f >= smallest_freq])

        return beam_vals

    # Get the deprojected power-spectrum

    pspec_dep = PowerSpectrum(deproj_img_hdu,
                              distance=fitinfo_dict[gal]['distance'])
    pspec_dep.run(verbose=False, beam_correct=False, fit_2D=False,
                  high_cut=0.1 / u.pix,
                  use_pyfftw=True, threads=ncores,
                  apodize_kernel=fitinfo_dict[gal]['apod_kern'])

    # Now fit both power-spectra

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    if fitinfo_dict[gal]['high_cut'] is not None:
        high_cut = fitinfo_dict[gal]['high_cut']

    else:
        # high_cut = (1 / (beam_gauss_width * 5.))
        high_cut = (1 / (beam_gauss_width * 1.5))

    # Fit on scales > 3 pixels to avoid flattening from pixelization
    # fit_mask = pspec.freqs.value < 1 / 3.
    # fit_mask = pspec.freqs.value < 0.1
    fit_mask = pspec.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    fit_mask = np.logical_and(fit_mask, np.isfinite(pspec.ps1D))
    fit_mask = np.logical_and(fit_mask, np.isfinite(pspec.ps1D_stddev))

    freqs = pspec.freqs.value[fit_mask]
    ps1D = pspec.ps1D[fit_mask]
    ps1D_stddev = pspec.ps1D_stddev[fit_mask]
    # ps1D_stddev = np.ones_like(ps1D_stddev)

    def beam_model_orig(f):
        return gaussian_beam(f, beam_gauss_width)

    fixB = False
    noise_term = True

    out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                  ps1D_stddev,
                                                  beam_model=beam_model_orig,
                                                  nsamp=nsamp,
                                                  fixB=fixB,
                                                  noise_term=noise_term)

    row_names.append("{}_orig".format(gal.lower()))

    fit_results['logA'].append(np.array(summ['mean'])[0])
    fit_results['ind'].append(np.array(summ['mean'])[1])

    if fixB:
        fit_results['logB'].append(-20)
    else:
        fit_results['logB'].append(np.array(summ['mean'])[2])

    if noise_term:
        fit_results['logC'].append(np.array(summ['mean'])[-1])
    else:
        fit_results['logC'].append(-20)

    fit_results['logA_std'].append(np.array(summ['sd'])[0])
    fit_results['ind_std'].append(np.array(summ['sd'])[1])
    if fixB:
        fit_results['logB_std'].append(0.)
    else:
        fit_results['logB_std'].append(np.array(summ['sd'])[2])

    if noise_term:
        fit_results['logC_std'].append(np.array(summ['sd'])[-1])
    else:
        fit_results['logC_std'].append(0.)

    print("Orig Fit params: {}".format(out[0]))
    print("Orig Fit errs: {}".format(out[1]))

    # Now fit the deproj version

    beam_size_dep = pspec_dep._beam.major.to(u.deg) / pspec_dep._ang_size.to(u.deg)
    beam_size_dep = beam_size_dep.value
    beam_gauss_width_dep = beam_size_dep / np.sqrt(8 * np.log(2))

    if fitinfo_dict[gal]['high_cut'] is not None:
        high_cut_dep = fitinfo_dict[gal]['high_cut']

    else:
        # high_cut_dep = (1 / (beam_gauss_width_dep * 3.))
        high_cut_dep = (1 / (beam_gauss_width_dep * 5.))

    fit_mask_dep = pspec_dep.freqs.value < high_cut_dep

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask_dep[:2] = False

    freqs_dep = pspec_dep.freqs.value[fit_mask_dep]
    ps1D_dep = pspec_dep.ps1D[fit_mask_dep]
    ps1D_stddev_dep = pspec_dep.ps1D_stddev[fit_mask_dep]

    out_dep, summ_dep, trace_dep, fit_model_dep = \
        fit_pspec_model(freqs_dep, ps1D_dep,
                        ps1D_stddev_dep,
                        beam_model=beam_model_dep,
                        nsamp=nsamp,
                        fixB=fixB,
                        noise_term=noise_term)

    row_names.append("{}_dep".format(gal.lower()))

    fit_results['logA'].append(np.array(summ_dep['mean'])[0])
    fit_results['ind'].append(np.array(summ_dep['mean'])[1])

    if fixB:
        fit_results['logB'].append(-20)
    else:
        fit_results['logB'].append(np.array(summ_dep['mean'])[2])

    if noise_term:
        fit_results['logC'].append(np.array(summ_dep['mean'])[-1])
    else:
        fit_results['logC'].append(-20)

    fit_results['logA_std'].append(np.array(summ_dep['sd'])[0])
    fit_results['ind_std'].append(np.array(summ_dep['sd'])[1])
    if fixB:
        fit_results['logB_std'].append(0.)
    else:
        fit_results['logB_std'].append(np.array(summ_dep['sd'])[2])

    if noise_term:
        fit_results['logC_std'].append(np.array(summ_dep['sd'])[-1])
    else:
        fit_results['logC_std'].append(0.)

    print("Dep Fit params: {}".format(out_dep[0]))
    print("Dep Fit errs: {}".format(out_dep[1]))

    # Make 2 plots:

    # (1) Trace plot for the deprojected fit (the orig column density is already
    # fit in fit_pspec_coldens.py).

    tr_plot = pm.traceplot(trace_dep)

    plot_savename = osjoin(plot_folder, "{0}_coldens.deproj.pspec_wbeam_traceplot.png".format(gal))

    plt.savefig(plot_savename)

    plot_savename = osjoin(plot_folder, "{0}_coldens.deproj.pspec_wbeam_traceplot.png".format(gal))
    plt.savefig(plot_savename)

    plt.close()

    # (2) Both power-spectra with their respective fits.

    plt.figure(figsize=(4.2, 2.9))

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    phys_freqs_dep = pspec_dep._spatial_freq_unit_conversion(pspec_dep.freqs, u.pc**-1).value

    phys_scales_dep = 1 / phys_freqs_dep

    plt.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10,
               label='Original')
    plt.loglog(phys_scales_dep, pspec_dep.ps1D, 'k--', zorder=-10,
               label='Deprojected')

    # beam_amp = 10**(max(out[0][0], out[0][2]) - 1.)

    logA = fit_results['logA'][-2]
    ind = fit_results['ind'][-2]
    pars = [logA, ind]

    if fixB:
        pars.append(-20)
    else:
        logB = fit_results['logB'][-2]
        pars.append(logB)

    if noise_term:
        logC = fit_results['logC'][-2]
        pars.append(logC)

    plt.loglog(phys_scales[fit_mask],
               fit_model(freqs, *pars), 'r-',
               linewidth=3,)


    logA = fit_results['logA'][-1]
    ind = fit_results['ind'][-1]
    pars = [logA, ind]

    if fixB:
        pars.append(-20)
    else:
        logB = fit_results['logB'][-1]
        pars.append(logB)

    if noise_term:
        logC = fit_results['logC'][-1]
        pars.append(logC)

    plt.loglog(phys_scales_dep[fit_mask_dep],
               fit_model_dep(freqs_dep, *pars), 'r--',
               linewidth=3,)
    # plt.loglog(phys_scales[fit_mask],
    #            beam_amp * beam_model(freqs), 'r:', label='PSF')

    plt.legend(frameon=True, loc='upper right')

    # Also plot a set of 10 random parameter draws

    phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

    plt.axvline(1 / phys_beam, linestyle='-', linewidth=4,
                alpha=0.8, color='gray')

    # phys_beam_dep = pspec_dep._spatial_freq_unit_conversion(1 / (beam_size_dep * u.pix), u.pc**-1).value

    # plt.axvline(1 / phys_beam_dep, linestyle='--', linewidth=4,
    #             alpha=0.8, color='gray')

    plt.grid()

    plt.xlabel(r"Scale (pc)")

    plt.gca().invert_xaxis()

    plt.tight_layout()

    plot_savename = osjoin(plot_folder, "{0}_coldens_deprojcompare.1Dpspec_wbeam.png".format(gal))
    plt.savefig(plot_savename)
    plot_savename = osjoin(plot_folder, "{0}_coldens_deprojcompare.1Dpspec_wbeam.pdf".format(gal))
    plt.savefig(plot_savename)

    plt.close()

df = pd.DataFrame(fit_results, index=row_names)
df.to_csv(osjoin(data_path, "pspec_coldens_deproj_fit_results.csv"))
