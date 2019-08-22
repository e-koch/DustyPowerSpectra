
'''
Compute the power-spectra for the Mag clouds, M31, and M33
with the dust column density, and the individual IR maps.

Fitting doesn't matter here. We'll save the classes to do
a thorough job fitting later.
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

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays

make_interactive = False

if not plt.isinteractive() and make_interactive:
    plt.ion()
else:
    plt.ioff()

osjoin = os.path.join

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/IC342/")
otherdata_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
plot_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/IC342_plots")

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

run_pspec = False
run_pspec_psfs = False
fit_pspec = True

gal = 'IC342'
dist = 3400 * u.kpc
res_types = ['orig']

if run_pspec:

    fitinfo_dict = dict()

    fitinfo_dict["IC342"] = \
        {'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                     'filename': "IC0342_scanamorphos_v16.9_pacs100_0.fits",
                     'low_int_cut': None, 'high_int_cut': None},
         'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': None,
                     'filename': "IC0342_scanamorphos_v16.9_pacs160_0.fits",
                     'low_int_cut': None, 'high_int_cut': None},
         'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire250_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None},
         'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire350_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None},
         'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire500_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None}}

    ncores = 1

    img_view = True

    skip_check = True

    # Load in the dust column density maps to set the allowed
    # spatial region

    filename_coldens = glob(osjoin(data_path, '455pc', "dust.fits"))

    hdu_coldens = fits.open(filename_coldens[0])

    pad_size = 0.5 * u.arcmin

    proj_coldens = Projection.from_hdu(fits.PrimaryHDU(hdu_coldens[0].data,
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

        for res_type in res_types:

            print("Resolution {}".format(res_type))

            filename = fitinfo_dict[gal][name]['filename']

            if not os.path.exists(osjoin(data_path, 'raw', filename)):
                print("Could not find {}. Skipping".format(filename))
                continue

            hdu = fits.open(osjoin(data_path, 'raw', filename))

            # Multiple images in the PACS imgs
            if 'pacs' in name:
                data = hdu[0].data[0]
            else:
                data = hdu[0].data

            proj = Projection.from_hdu(fits.PrimaryHDU(data,
                                                       hdu[0].header))
            # Attach equiv Gaussian beam
            if res_type == 'orig':
                proj = proj.with_beam(fitinfo_dict[gal][name]['beam'])
            # The convolved images should have all have a beam saved

            # Take minimal shape. Remove empty space.
            # Erode edges to avoid noisier region/uneven scans
            mask = np.isfinite(proj)
            mask = nd.binary_erosion(mask, np.ones((3, 3)), iterations=8)

            # Add radial box cut
            # radius = gal_cls.radius(header=proj.header).to(u.kpc)
            # rad_mask = radius < cut
            # mask = np.logical_and(mask, rad_mask)

            # Pick out region determined from the column density map extents
            # PLus some padding at the edge
            spat_mask = spat_mask_maker(*proj.spatial_coordinate_map)

            proj = proj[nd.find_objects(mask & spat_mask)[0]]

            # Save the cut-out, if it doesn't already exist
            out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))

            if not os.path.exists(osjoin(data_path, 'raw', out_filename)):
                proj.write(osjoin(data_path, 'raw', out_filename))

            # look at each image.
            if img_view:
                proj.quicklook()
                plt.draw()
                input("{}".format(filename))
                plt.close()

            if res_type == 'orig':
                save_name = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)
            else:
                save_name = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(), name, res_type)

            # For now skip already saved power-spectra
            if os.path.exists(osjoin(data_path, 'raw', save_name)) and skip_check:
                print("Already saved pspec for {}. Skipping".format(filename))
                continue
            else:
                os.system("rm -f {}".format(osjoin(data_path, 'raw', save_name)))

            pspec = PowerSpectrum(proj, distance=dist)
            pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                      high_cut=0.1 / u.pix,
                      use_pyfftw=True, threads=ncores,
                      apodize_kernel=fitinfo_dict[gal][name]['apod_kern'])

            pspec.save_results(osjoin(data_path, 'raw', save_name),
                               keep_data=False)

            del pspec, proj, hdu

# Make PSF power-spectra in with the same pix scale as the data
if run_pspec_psfs:

    from glob import glob
    import numpy as np
    from astropy.io import fits
    from spectral_cube import Projection
    from radio_beam import Beam
    import astropy.units as u
    from astropy.wcs.utils import proj_plane_pixel_scales
    import matplotlib.pyplot as plt

    from photutils import resize_psf, CosineBellWindow

    fitinfo_dict = dict()

    fitinfo_dict["IC342"] = \
        {'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                     'filename': "IC0342_scanamorphos_v16.9_pacs100_0.fits",
                     'low_int_cut': None, 'high_int_cut': None},
         'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': None,
                     'filename': "IC0342_scanamorphos_v16.9_pacs160_0.fits",
                     'low_int_cut': None, 'high_int_cut': None},
         'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire250_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None},
         'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire350_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None},
         'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
                      'filename': "IC0342_kingfish_spire500_v3-0_scan.fits",
                      'low_int_cut': None, 'high_int_cut': None}}

    names = {'pacs100': ["PSF_PACS_100.fits", 0],
             'pacs160': ["PSF_PACS_160.fits", 0],
             'spire250': ["PSF_SPIRE_250.fits", 1],
             'spire350': ["PSF_SPIRE_350.fits", 1],
             'spire500': ["PSF_SPIRE_500.fits", 1]}

    # Some images are large. Run fft in parallel
    ncores = 1

    skip_check = False

    kern_path = os.path.expanduser("~/bigdata/ekoch/Aniano_kernels/")

    for name in names:

        print("On {}".format(name))

        filename = fitinfo_dict[gal][name]['filename']

        hdu = fits.open(osjoin(data_path, 'raw', filename))

        # Multiple images in the PACS imgs
        if 'pacs' in name:
            data = hdu[0].data[0]
        else:
            data = hdu[0].data

        proj = Projection.from_hdu(fits.PrimaryHDU(data,
                                                   hdu[0].header))

        # Now open the kernel file
        kernfits_name = names[name][0]
        kernfits_ext = names[name][1]

        kernel_filename = osjoin(kern_path, kernfits_name)

        kern_proj = Projection.from_hdu(fits.open(osjoin(kern_path, kernel_filename))[kernfits_ext])

        img_scale = np.abs(proj_plane_pixel_scales(proj.wcs))[0]
        kern_scale = np.abs(proj_plane_pixel_scales(kern_proj.wcs))[0]

        kernel = resize_psf(kern_proj.value, kern_scale, img_scale)

        # Normalize to make a kernel
        kernel /= kernel.sum()

        kern_pspec = PowerSpectrum((kernel, kern_proj.header))
        kern_pspec.run(verbose=False, fit_2D=False)

        save_name = "{0}_kernel_{1}.pspec.pkl".format(name, gal.lower())

        kern_pspec.save_results(osjoin(data_path, 'raw', save_name),
                                keep_data=True)


if fit_pspec:

    fitinfo_dict = dict()

    fitinfo_dict["IC342"] = \
        {'pacs100': {'beam': Beam(7.1 * u.arcsec),
                     'low_cut': None, 'high_cut': None,
                     'use_beam': True, 'fixB': True},
         'pacs160': {'beam': Beam(11.2 * u.arcsec),
                     'low_cut': None, 'high_cut': None,
                     'use_beam': True, 'fixB': True},
         'spire250': {'beam': Beam(18.2 * u.arcsec),
                      'low_cut': None, 'high_cut': None,
                      'use_beam': True, 'fixB': True},
         'spire350': {'beam': Beam(25 * u.arcsec),
                      'low_cut': None, 'high_cut': None,
                      'use_beam': True, 'fixB': True},
         'spire500': {'beam': Beam(36.4 * u.arcsec),
                      'low_cut': None, 'high_cut': None,
                      'use_beam': True, 'fixB': True}}

    # Load model functions
    repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
    code_name = os.path.join(repo_path, "models.py")
    exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

    # For output of fits results
    fit_results = {'logA': [], 'ind': [],  # 'logB': [],
                   'logA_std': [], 'ind_std': []}  # , 'logB_std': []}
    row_names = []

    for name in fitinfo_dict[gal]:

        for res_type in res_types:

            print("Resolution {}".format(res_type))

            if res_type == 'orig':
                filename = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)
            else:
                filename = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(), name, res_type)

            if not os.path.exists(osjoin(data_path, 'raw', filename)):
                print("Could not find {}. Skipping".format(filename))
                continue

            # Load pspec object
            pspec = PowerSpectrum.load_results(osjoin(data_path, 'raw', filename))

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

            # if we're dealing with the original data, load in the saved power
            # spectrum of the normalized PSF
            if res_type == 'orig':
                kern_save_name = "{0}_kernel_{1}.pspec.pkl".format(name, gal.lower())

                kern_fpath = osjoin(data_path, 'raw', kern_save_name)
                if not os.path.exists(kern_fpath):
                    raise OSError("Pspec {0} not found.".format(kern_fpath))

                beam_model = make_psf_beam_function(kern_fpath)

            # otherwise use a gaussian beam model
            else:

                def beam_model(f):
                    return gaussian_beam(f, beam_gauss_width)

            nsamp = 6000

            # Set whether to use the beam_model
            if fitinfo_dict[gal][name]['use_beam']:
                fit_beam_model = beam_model
            else:
                fit_beam_model = None

            out, summ, trace, fit_model = \
                fit_pspec_model(freqs, ps1D,
                                ps1D_stddev,
                                beam_model=fit_beam_model,
                                nsamp=nsamp,
                                fixB=fitinfo_dict[gal][name]['fixB'])

            row_names.append("{0}_{1}_{2}".format(gal.lower(), name, res_type))

            fit_results['logA'].append(np.array(summ['mean'])[0])
            fit_results['ind'].append(np.array(summ['mean'])[1])
            # fit_results['logB'].append(np.array(summ['mean'])[2])

            fit_results['logA_std'].append(np.array(summ['sd'])[0])
            fit_results['ind_std'].append(np.array(summ['sd'])[1])
            # fit_results['logB_std'].append(np.array(summ['sd'])[2])

            plt.figure(figsize=(8.4, 2.9))

            plt.subplot(121)
            # plt.title("Fit_params: {}".format(out[0]))
            plt.loglog(pspec.freqs.value, pspec.ps1D, 'k', zorder=-10)

            # beam_amp = 10**(max(out[0][0], out[0][2]) - 1.)
            beam_amp = 10**(out[0][0] - 1.)

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
                # logB = trace.get_values('logB')[rint]

                # pars = np.array([logA, ind, logB])
                pars = np.array([logA, ind, -20.])

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

            plot_savename = osjoin(plot_path, "{0}.pspec_wbeam.png".format(filename.rstrip(".fits")))

            print(plot_savename)
            print("Fit_params: {}".format(out[0]))
            # print("Fit_errs: {}".format(np.sqrt(np.abs(np.diag(out[1])))))
            print("Fit_errs: {}".format(out[1]))
            if make_interactive:
                input("?")

            plt.savefig(plot_savename)

            plot_savename = osjoin(plot_path, "{0}.pspec_wbeam.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            tr_plot = pm.traceplot(trace)

            plot_savename = osjoin(plot_path, "{0}.pspec_wbeam_traceplot.png".format(filename.rstrip(".fits")))

            plt.draw()
            if make_interactive:
                input("?")

            plt.savefig(plot_savename)

            plot_savename = osjoin(plot_path, "{0}.pspec_wbeam_traceplot.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            # OneD spectrum by itself

            plt.figure(figsize=(4.2, 2.9))

            phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

            phys_scales = 1 / phys_freqs

            plt.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

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
                # logB = trace.get_values('logB')[rint]

                # pars = np.array([logA, ind, logB])
                pars = np.array([logA, ind, -20.])

                rand_pars.append(pars)

                plt.loglog(phys_scales[fit_mask],
                           fit_model(freqs, *pars),
                           color='gray', alpha=0.25,
                           linewidth=3, zorder=-1)

            # Save the random samples to a npy file
            randfilename = osjoin(data_path, 'raw',
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

            plot_savename = osjoin(plot_path, "{0}.1Dpspec_wbeam.png".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)
            plot_savename = osjoin(plot_path, "{0}.1Dpspec_wbeam.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

    df = pd.DataFrame(fit_results, index=row_names)
    df.to_csv(os.path.join(data_path, "IC342_pspec_fit_results.csv"))
