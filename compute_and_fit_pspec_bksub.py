
'''
Compute and fit the power spectra of the Gaussian-PSF background subtracted
images.
'''

import os
from glob import glob
import numpy as np
from astropy.io import fits
from spectral_cube import Projection
from scipy import ndimage as nd
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sb
from radio_beam import Beam
import pymc3 as pm
import pandas as pd

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


make_interactive = False

if not plt.isinteractive() and make_interactive:
    plt.ion()
else:
    plt.ioff()

    # Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Load model functions
repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
# from fit_settings import fitinfo_dict

run_pspec = False
run_fits = False
run_summary_plot = False

skip_check = False

ncores = 5

fitinfo_dict = dict()

fitinfo_dict["LMC"] = \
    {'mips24': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': False},
     'mips70': {'beam': Beam(30.0 * u.arcsec), 'apod_kern': None,
                'low_int_cut': -1000., 'high_int_cut': 4000,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': False},
     'pacs100': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': False},
     'mips160': {'beam': Beam(64 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': False},
     'pacs160': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': False},
     'spire250': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire350': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire500': {'beam': Beam(41 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True}}

fitinfo_dict["SMC"] = \
    {'mips24': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': False},
     'mips70': {'beam': Beam(30.0 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': 2000,
                'low_cut': None, 'high_cut': 0.06,
                'use_beam': False, 'fixB': False},
     'pacs100': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': -1000., 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': False},
     'mips160': {'beam': Beam(64 * u.arcsec), 'apod_kern': 'tukey',
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': False},
     'pacs160': {'beam': Beam(15 * u.arcsec), 'apod_kern': 'tukey',
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': 0.1,
                 'use_beam': False, 'fixB': True},
     'spire250': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': False},
     'spire350': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': False},
     'spire500': {'beam': Beam(41 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': False}}

fitinfo_dict["M33"] = \
    {'mips24': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': True},
     'mips70': {'beam': Beam(30.0 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': True},
     'pacs100': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': True},
     'mips160': {'beam': Beam(64 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': True},
     'pacs160': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': True},
     'spire250': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire350': {'beam': Beam(30 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire500': {'beam': Beam(41 * u.arcsec), 'apod_kern': None,
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True}}

fitinfo_dict["M31"] = \
    {'mips24': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': False},
     'mips70': {'beam': Beam(30.0 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None,
                'low_cut': None, 'high_cut': None,
                'use_beam': True, 'fixB': False},
     'pacs100': {'beam': Beam(15 * u.arcsec), 'apod_kern': None,
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': 0.03,
                 'use_beam': False, 'fixB': True},
     'mips160': {'beam': Beam(64 * u.arcsec), 'apod_kern': 'tukey',
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': True},
     'pacs160': {'beam': Beam(15 * u.arcsec), 'apod_kern': 'tukey',
                 'low_int_cut': None, 'high_int_cut': None,
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True, 'fixB': True},
     'spire250': {'beam': Beam(30 * u.arcsec), 'apod_kern': 'tukey',
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire350': {'beam': Beam(30 * u.arcsec), 'apod_kern': 'tukey',
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True},
     'spire500': {'beam': Beam(41 * u.arcsec), 'apod_kern': 'tukey',
                  'low_int_cut': None, 'high_int_cut': None,
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True, 'fixB': True}}

gals = {'LMC': 50.1 * u.kpc, 'SMC': 62.1 * u.kpc,
        'M33': 840 * u.kpc, 'M31': 744 * u.kpc}

if run_pspec:
    for gal in gals:

        dist = gals[gal]

        # Load in the dust column density maps to set the allowed
        # spatial region

        filename_coldens = glob(osjoin(data_path, gal, "*dust.surface.density*.fits"))

        hdu_coldens = fits.open(filename_coldens[0])

        pad_size = 0.5 * u.arcmin

        proj_coldens = Projection.from_hdu(fits.PrimaryHDU(hdu_coldens[0].data[0].squeeze(),
                                                           hdu_coldens[0].header))

        # Get minimal size
        proj_coldens = proj_coldens[nd.find_objects(np.isfinite(proj_coldens))[0]]

        # Get spatial extents.
        # NOTE: extrema for 2D objects broken in spectral-cube! Need to fix...
        lat, lon = proj_coldens.spatial_coordinate_map
        lat_min = lat.min() - pad_size
        lat_max = lat.max() + pad_size
        lon_min = lon.min() - pad_size
        lon_max = lon.max() + pad_size

        def spat_mask_maker(lat_map, lon_map):

            lat_mask = np.logical_and(lat_map > lat_min,
                                      lat_map < lat_max)
            lon_mask = np.logical_and(lon_map > lon_min,
                                      lon_map < lon_max)
            return lat_mask & lon_mask

        print("On {}".format(gal))

        for name in fitinfo_dict[gal]:

            print("On {}".format(name))

            beamsize_str = f"{int(fitinfo_dict[gal][name]['beam'].major.value)}"

            filename = f"{gal.lower()}_{name}_gauss{beamsize_str}_bksub.fits"

            if not os.path.exists(osjoin(data_path, gal, filename)):
                print("Could not find {}. Skipping".format(filename))
                continue

            hdu = fits.open(osjoin(data_path, gal, filename))
            proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                       hdu[0].header))

            proj = proj.with_beam(fitinfo_dict[gal][name]['beam'])

            # Take minimal shape. Remove empty space.
            # Erode edges to avoid noisier region/uneven scans
            mask = np.isfinite(proj)
            mask = nd.binary_erosion(mask, np.ones((3, 3)), iterations=8)

            spat_mask = spat_mask_maker(*proj.spatial_coordinate_map)

            proj = proj[nd.find_objects(mask & spat_mask)[0]]

            # Save the cut-out, if it doesn't already exist
            out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))

            if not os.path.exists(osjoin(data_path, gal, out_filename)):
                proj.write(osjoin(data_path, gal, out_filename))

            save_name = f"{out_filename.rstrip('.fits')}.pspec.pkl"

            # For now skip already saved power-spectra
            if os.path.exists(osjoin(data_path, gal, save_name)) and skip_check:
                print("Already saved pspec for {}. Skipping".format(filename))
                continue
            else:
                os.system("rm -f {}".format(osjoin(data_path, gal, save_name)))

            pspec = PowerSpectrum(proj, distance=dist)
            pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                      high_cut=0.1 / u.pix,
                      use_pyfftw=True, threads=ncores,
                      apodize_kernel=fitinfo_dict[gal][name]['apod_kern'])

            pspec.save_results(osjoin(data_path, gal, save_name),
                               keep_data=False)

            del pspec, proj, hdu


if run_fits:

    row_names = []
    fit_results = {'logA': [], 'ind': [], 'logB': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': []}

    for gal in gals:

        dist = gals[gal]

        # Make a plot output folder
        plot_folder = osjoin(data_path, "{}_plots".format(gal))
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        # Make a folder for the bksub plots
        plot_folder = osjoin(plot_folder, 'pspec_bksub_plots')
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        for name in fitinfo_dict[gal]:

            print("On {}".format(name))

            beamsize_str = f"{int(fitinfo_dict[gal][name]['beam'].major.value)}"

            filename = f"{gal.lower()}_{name}_gauss{beamsize_str}_bksub.fits"
            out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))
            pspec_save_name = f"{out_filename.rstrip('.fits')}.pspec.pkl"

            # Load pspec object
            pspec = PowerSpectrum.load_results(osjoin(data_path, gal, pspec_save_name))

            # Beam doesn't stay cached. Don't know why
            pspec.load_beam()



            beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
            beam_size = beam_size.value
            beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

            if fitinfo_dict[gal][name]['high_cut'] is not None:
                high_cut = fitinfo_dict[gal][name]['high_cut']

            else:
                high_cut = (1 / (beam_gauss_width * 3.))

            # Fit on scales > 3 pixels to avoid flattening from pixelization
            # fit_mask = pspec.freqs.value < 1 / 3.
            # fit_mask = pspec.freqs.value < 0.1
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

            # Set whether to use the beam_model
            if fitinfo_dict[gal][name]['use_beam']:
                fit_beam_model = beam_model
            else:
                fit_beam_model = None

            out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                          ps1D_stddev,
                                                          beam_model=fit_beam_model,
                                                          nsamp=nsamp)


            row_names.append("{0}_{1}".format(gal.lower(), name))

            fit_results['logA'].append(np.array(summ['mean'])[0])
            fit_results['ind'].append(np.array(summ['mean'])[1])
            fit_results['logB'].append(np.array(summ['mean'])[2])

            fit_results['logA_std'].append(np.array(summ['sd'])[0])
            fit_results['ind_std'].append(np.array(summ['sd'])[1])
            fit_results['logB_std'].append(np.array(summ['sd'])[2])

            # Make some plots
            plt.figure(figsize=(8.4, 2.9))

            plt.subplot(121)
            # plt.title("Fit_params: {}".format(out[0]))
            plt.loglog(pspec.freqs.value, pspec.ps1D, 'k', zorder=-10)

            beam_amp = 10**(max(out[0][0], out[0][2]) - 1.)

            plt.loglog(freqs, fit_model(freqs, *out[0]), 'r--',
                       linewidth=3, label='Fit')
            plt.loglog(freqs,
                       beam_amp * beam_model(freqs), 'r:', label='PSF')

            plt.xlabel("Freq. (1 / pix)")

            plt.legend(frameon=True, loc='upper right')

            # Also plot a set of 10 random parameter draws

            # Get some random draws
            randints = np.random.randint(0, high=nsamp, size=10)

            for rint in randints:
                logA = trace.get_values('logA')[rint]
                ind = trace.get_values('index')[rint]
                logB = trace.get_values('logB')[rint]

                pars = np.array([logA, ind, logB])

                plt.loglog(freqs, fit_model(freqs, *pars),
                           color='gray', alpha=0.2,
                           linewidth=3, zorder=-1)

            plt.axvline(1 / beam_size, linestyle=':', linewidth=4,
                        alpha=0.8, color='gray')
            # plt.axvline(1 / beam_gauss_width)

            plt.grid()

            plt.subplot(122)
            # plt.title(filename)
            plt.imshow(np.log10(pspec.ps2D), origin='lower', cmap='plasma')
            cbar = plt.colorbar()

            # Add contour showing region fit
            yy_freq, xx_freq = make_radial_freq_arrays(pspec.ps2D.shape)

            freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)

            mask = freqs_dist <= high_cut

            plt.contour(mask, colors=['k'], linestyles=['--'])

            plt.tight_layout()

            plt.draw()

            plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam.png".format(filename.rstrip(".fits")))

            print(plot_savename)
            print("Fit_params: {}".format(out[0]))
            # print("Fit_errs: {}".format(np.sqrt(np.abs(np.diag(out[1])))))
            print("Fit_errs: {}".format(out[1]))
            if make_interactive:
                input("?")


            plt.savefig(plot_savename)

            plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            tr_plot = pm.traceplot(trace)

            plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam_traceplot.png".format(filename.rstrip(".fits")))

            plt.draw()
            if make_interactive:
                input("?")

            plt.savefig(plot_savename)

            plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam_traceplot.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            # OneD spectrum by itself

            plt.figure(figsize=(4.2, 2.9))

            phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

            phys_scales = 1 / phys_freqs

            plt.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

            beam_amp = 10**(max(out[0][0], out[0][2]) - 1.)

            plt.loglog(phys_scales[fit_mask],
                       fit_model(freqs, *out[0]), 'r--',
                       linewidth=3, label='Fit')
            plt.loglog(phys_scales[fit_mask],
                       beam_amp * beam_model(freqs), 'r:', label='PSF')

            plt.legend(frameon=True, loc='upper right')

            # Also plot a set of 10 random parameter draws

            # Get some random draws
            randints = np.random.randint(0, high=nsamp, size=10)

            # Hang onto the random samples for the paper plots.
            rand_pars = []

            for rint in randints:
                logA = trace.get_values('logA')[rint]
                ind = trace.get_values('index')[rint]
                logB = trace.get_values('logB')[rint]

                pars = np.array([logA, ind, logB])

                rand_pars.append(pars)

                plt.loglog(phys_scales[fit_mask],
                           fit_model(freqs, *pars),
                           color='gray', alpha=0.25,
                           linewidth=3, zorder=-1)

            # Save the random samples to a npy file
            randfilename = osjoin(data_path, gal.upper(),
                                  f"{filename}_param_samples.npy")
            np.save(randfilename, np.array(rand_pars))

            phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

            plt.axvline(1 / phys_beam, linestyle=':', linewidth=4,
                        alpha=0.8, color='gray')
            # plt.axvline(1 / beam_gauss_width)

            plt.grid()

            # switch labels to spatial scale rather than frequency
            # ax1 = plt.gca()
            # ax1Xs = [r"$10^{}$".format(int(-ind)) for ind in np.log10(ax1.get_xticks())]
            # ax1.set_xticklabels(ax1Xs)

            plt.xlabel(r"Scale (pc)")

            plt.gca().invert_xaxis()

            plt.tight_layout()

            plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.png".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)
            plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

    df = pd.DataFrame(fit_results, index=row_names)
    df.to_csv(os.path.join(data_path, "pspec_fit_results_bksub.csv"))

if run_summary_plot:

    # Compare the fitted indices.
    # There is little/no effect on the power spectrum due to background
    # subtraction.

    df = pd.read_csv(osjoin(data_path, "pspec_fit_results.csv"))
    df_bksub = pd.read_csv(osjoin(data_path,
                                  "pspec_fit_results_bksub.csv"))

    plt.figure(figsize=(4.2, 2.9))

    plt.errorbar(np.arange(32), df['ind'][::2], yerr=df['ind_std'][::2],
                 linestyle='--', label='Orig. PSF No bksub.')
    plt.errorbar(np.arange(32), df['ind'][1::2], yerr=df['ind_std'][1::2],
                 linestyle=':', label='Gauss PSF No bksub.')
    plt.errorbar(np.arange(32), df_bksub['ind'], yerr=df_bksub['ind_std'],
                 linestyle='-', label='Gauss PSF Bksub.')

    plt.legend(loc='best', frameon=True)

    plt.xlabel("Photometric Bands for all Galaxies")
    plt.ylabel("Power Spectrum Index.")

    plt.grid()
    plt.tight_layout()

    plot_folder = osjoin(data_path, "summary_plots")
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    plt.savefig("pspec_index_bksub_comparison.pdf")
    plt.savefig("pspec_index_bksub_comparison.png")
    plt.close()
