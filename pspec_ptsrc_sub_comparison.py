
'''
Compare the SMC and LMC MIPS bands w/ and w/o pt. src subtraction.
'''

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.optimize import curve_fit
from radio_beam import Beam
import pymc3 as pm
import theano.tensor as T
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from glob import glob

make_interactive = True

do_savepspec = False
do_fitpspec = True

if not plt.isinteractive() and make_interactive:
    plt.ion()

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum

# Load in models
repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))


# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
# data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")


names = {'mips160': Beam(38.8 * u.arcsec),
         # 'mips24': Beam(6.5 * u.arcsec),  # No whole residual map for LMC
         'mips70': Beam(18.7 * u.arcsec)}

gals = ['LMC', 'SMC']

res_types = ['orig']

# dist_cuts = [5, 3, 12, 20] * u.kpc

distances = [50.1 * u.kpc, 62.1 * u.kpc]

# Some images are large. Run fft in parallel
ncores = 6

img_view = False

skip_check = True

make_interactive = False

if do_savepspec:

    for gal, dist in zip(gals, distances):

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

        for name in names:

            print("On {}".format(name))

            for res_type in res_types:

                print("Resolution {}".format(res_type))

                filename = "{0}_{1}_mjysr_residual.fits".format(gal.lower(), name)

                if not os.path.exists(osjoin(data_path, gal, filename)):
                    print("Could not find {}. Skipping".format(filename))
                    continue

                hdu = fits.open(osjoin(data_path, gal, filename))
                proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                           hdu[0].header))
                # Attach equiv Gaussian beam
                if res_type == 'orig':
                    proj = proj.with_beam(names[name])
                # The convolved images should have all have a beam saved

                # Take minimal shape. Remove empty space.
                # Erode edges to avoid noisier region/uneven scans
                mask = np.isfinite(proj)
                mask = nd.binary_erosion(mask, np.ones((3, 3)), iterations=8)

                # Pick out region determined from the column density map extents
                # PLus some padding at the edge
                spat_mask = spat_mask_maker(*proj.spatial_coordinate_map)

                proj = proj[nd.find_objects(mask & spat_mask)[0]]

                # Save the cut-out, if it doesn't already exist
                out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))

                if not os.path.exists(osjoin(data_path, gal, out_filename)):
                    proj.write(osjoin(data_path, gal, out_filename))

                # look at each image.
                if img_view:
                    proj.quicklook()
                    plt.draw()
                    input("{}".format(filename))
                    plt.close()

                save_name = "{0}_{1}_mjysr_residual.pspec.pkl".format(gal.lower(), name)

                # For now skip already saved power-spectra
                if os.path.exists(osjoin(data_path, gal, save_name)) and skip_check:
                    print("Already saved pspec for {}. Skipping".format(filename))
                    continue
                else:
                    os.system("rm -f {}".format(osjoin(data_path, gal, save_name)))

                pspec = PowerSpectrum(proj, distance=dist)
                pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                          high_cut=0.1 / u.pix,
                          use_pyfftw=True, threads=ncores)

                pspec.save_results(osjoin(data_path, gal, save_name),
                                   keep_data=False)

                del pspec, proj, hdu

if do_fitpspec:


    fit_results = {'logA': [], 'ind': [], 'logB': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': []}
    row_names = []

    for gal, dist in zip(gals, distances):

        print("On {}".format(gal))

        # Make a plot output folder
        plot_folder = osjoin(data_path, "{}_plots".format(gal))
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        for name in names:

            print("On {}".format(name))

            for res_type in res_types:

                print("Resolution {}".format(res_type))

                filename = "{0}_{1}_mjysr_residual.pspec.pkl".format(gal.lower(), name)

                if not os.path.exists(osjoin(data_path, gal, filename)):
                    print("Could not find {}. Skipping".format(filename))
                    continue

                # Load pspec object
                pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))

                # Beam doesn't stay cached. Don't know why
                pspec.load_beam()

                beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
                beam_size = beam_size.value
                beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

                # Fit on scales > 3 pixels to avoid flattening from pixelization
                # fit_mask = pspec.freqs.value < 1 / 3.
                fit_mask = pspec.freqs.value < (1 / (beam_gauss_width * 3.))
                # fit_mask = pspec.freqs.value < 0.1

                # And cut out the largest scales due to expected deviations with
                # small stddev
                fit_mask[:2] = False

                freqs = pspec.freqs.value[fit_mask]
                ps1D = pspec.ps1D[fit_mask]
                ps1D_stddev = pspec.ps1D_stddev[fit_mask]

                # if we're dealing with the original data, load in the saved power
                # spectrum of the normalized PSF
                kern_save_name = "{0}_kernel_{1}.pspec.pkl".format(name, gal.lower())

                kern_fpath = osjoin(data_path, gal, kern_save_name)
                if not os.path.exists(kern_fpath):
                    raise OSError("Pspec {0} not found.".format(kern_fpath))

                beam_model = make_psf_beam_function(kern_fpath)

                nsamp = 6000

                out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                              ps1D_stddev,
                                                              beam_model=beam_model,
                                                              nsamp=nsamp)

                row_names.append("{0}_{1}_{2}".format(gal.lower(), name, res_type))

                fit_results['logA'].append(np.array(summ['mean'])[0])
                fit_results['ind'].append(np.array(summ['mean'])[1])
                fit_results['logB'].append(np.array(summ['mean'])[2])

                fit_results['logA_std'].append(np.array(summ['sd'])[0])
                fit_results['ind_std'].append(np.array(summ['sd'])[1])
                fit_results['logB_std'].append(np.array(summ['sd'])[2])

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

                    plt.loglog(freqs, powerlaw_model_wbeam(freqs, *pars),
                               color='gray', alpha=0.2,
                               linewidth=3, zorder=-1)


                plt.axvline(1 / beam_size, linestyle=':', linewidth=4,
                            alpha=0.8, color='gray')
                # plt.axvline(1 / beam_gauss_width)

                plt.grid()

                plt.subplot(122)
                # plt.title(filename)
                plt.imshow(np.log10(pspec.ps2D), origin='lower')
                plt.colorbar()

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

                plt.loglog(phys_freqs, pspec.ps1D, 'k', zorder=-10)

                beam_amp = 10**(max(out[0][0], out[0][2]) - 1.)

                plt.loglog(phys_freqs[fit_mask],
                           fit_model(freqs, *out[0]), 'r--',
                           linewidth=3, label='Fit')
                plt.loglog(phys_freqs[fit_mask],
                           beam_amp * beam_model(freqs), 'r:', label='PSF')

                plt.xlabel(r"Freq. (pc$^{-1}$)")

                plt.legend(frameon=True, loc='upper right')

                # Also plot a set of 10 random parameter draws

                # Get some random draws
                randints = np.random.randint(0, high=nsamp, size=10)

                for rint in randints:
                    logA = trace.get_values('logA')[rint]
                    ind = trace.get_values('index')[rint]
                    logB = trace.get_values('logB')[rint]

                    pars = np.array([logA, ind, logB])

                    plt.loglog(phys_freqs[fit_mask],
                               fit_model(freqs, *pars),
                               color='gray', alpha=0.25,
                               linewidth=3, zorder=-1)

                phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

                plt.axvline(phys_beam, linestyle=':', linewidth=4,
                            alpha=0.8, color='gray')
                # plt.axvline(1 / beam_gauss_width)

                plt.grid()

                plt.tight_layout()

                plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.png".format(filename.rstrip(".fits")))
                plt.savefig(plot_savename)
                plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.pdf".format(filename.rstrip(".fits")))
                plt.savefig(plot_savename)

                plt.close()

            # plt.draw()
            # print(out[0])
            # input(filename)
            # plt.clf()

    df = pd.DataFrame(fit_results, index=row_names)
    df.to_csv(os.path.join(data_path, "pspec_mips_ptsrc_sub_fit_results.csv"))

