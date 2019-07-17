
'''
Split the Mag Cloud column density maps into region a few beams across,
calculate the power-spectrum to look for variations.
'''

import numpy as np
from itertools import product


def get_coverage(image, spacing):
    """
    Returns locations of SAAss

    Borrowed from scouse: https://github.com/jdhenshaw/scousepy

    """
    cols, rows = np.where(np.isfinite(image))

    rangex = [np.min(rows), np.max(rows)]
    sizex = np.abs(np.min(rows) - np.max(rows))
    rangey = [np.min(cols), np.max(cols)]
    sizey = np.abs(np.min(cols) - np.max(cols))

    nposx = int((sizex / (spacing)) + 1.0)
    nposy = int((sizey / (spacing)) + 1.0)

    cov_x = (np.max(rangex) - (spacing) * np.arange(nposx)).astype(int)
    cov_y = (np.min(rangey) + (spacing) * np.arange(nposy)).astype(int)

    posns = np.array(list(product(cov_y, cov_x)))
    return posns


if __name__ == '__main__':

    from spectral_cube import Projection
    from astropy.io import fits
    import astropy.units as u
    import matplotlib.pyplot as plt
    import seaborn as sb
    from astropy.utils.console import ProgressBar
    import os
    from astropy.wcs import WCS
    from radio_beam import Beam
    import scipy.ndimage as nd
    import pandas as pd
    import logging

    from turbustat.statistics import PowerSpectrum
    from turbustat.statistics.psds import make_radial_freq_arrays

    osjoin = os.path.join

    make_interactive = False

    if make_interactive:
        plt.ion()
    else:
        plt.ioff()
        # Disable logging to screen to avoid hanging when closing x2go session
        logger = logging.getLogger('pymc3')
        logger.setLevel(logging.ERROR)

    # Load model functions
    repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
    code_name = os.path.join(repo_path, "models.py")
    exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

    # Running on SegFault w/ data on bigdata
    data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
    # data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

    fitinfo_dict = dict()

    fitinfo_dict["LMC"] = {'filename': r"lmc_dust.surface.density_FB.beta=1.8_gauss53.4_regrid_mwsub.fits",
                           'beam': Beam(53.4 * u.arcsec), 'apod_kern': 'tukey',
                           'low_int_cut': None, 'high_int_cut': None,
                           'low_cut': None, 'high_cut': None,
                           'distance': 50.1 * u.kpc}
    fitinfo_dict["SMC"] = {'filename': r"smc_dust.surface.density_FB.beta=1.8_gauss43.2_regrid_mwsub.fits",
                           'beam': Beam(43.2 * u.arcsec), 'apod_kern': 'tukey',
                           'low_int_cut': None, 'high_int_cut': None,
                           'low_cut': None, 'high_cut': None,
                           'distance': 62.1 * u.kpc}

    # Size of cut-outs
    cutout_size = 80

    # Fraction attenuated by apodizing kernel
    alpha = 0.1

    # Center location between minimal overlap: sqrt(2) * (1-alpha) * cutout_size
    # Or at the attenuated radius: (1 - alpha) * cutout_size / sqrt(2)
    spacing = np.ceil((1 - alpha) * cutout_size / np.sqrt(2)).astype(int)

    ncores = 1

    # nsamp = 1000
    nsamp = 6000

    row_names = []

    for gal in fitinfo_dict:

        print("On {}".format(gal))

        filename = osjoin(data_path, gal, fitinfo_dict[gal]['filename'])

        hdu_coldens = fits.open(filename)

        # The edges of the maps have high uncertainty. For M31, this may be altering
        # the shape of the power-spectrum. Try removing these edges:
        coldens_mask = np.isfinite(hdu_coldens[0].data[0].squeeze())

        coldens_mask = nd.binary_erosion(coldens_mask,
                                         structure=np.ones((3, 3)),
                                         iterations=8)

        # Get minimal size
        masked_data = hdu_coldens[0].data[0].squeeze()
        masked_data[~coldens_mask] = np.NaN
        proj_coldens = Projection.from_hdu(fits.PrimaryHDU(masked_data,
                                                           hdu_coldens[0].header))

        proj_coldens = proj_coldens[nd.find_objects(coldens_mask)[0]]

        proj_coldens = proj_coldens.with_beam(fitinfo_dict[gal]['beam'])

        mom0 = proj_coldens

        # Center locations
        posns = get_coverage(mom0.value, spacing)

        plot_folder = osjoin(data_path, "{}_plots".format(gal))
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        perposn_figfolder = os.path.join(plot_folder, "pspec_coldens_perpoint")
        if not os.path.exists(perposn_figfolder):
            os.mkdir(perposn_figfolder)

        # Save folder for power-spectra
        pspec_posn_folder = osjoin(data_path, gal, 'pspec_coldens_perpoint')
        if not os.path.exists(pspec_posn_folder):
            os.mkdir(pspec_posn_folder)

        # default_figure()

        fit_results = {'logA': [], 'ind': [], 'logB': [], 'logC': [],
                       'logA_std': [], 'ind_std': [], 'logB_std': [],
                       'logC_std': [], 'x': [], 'y': [], 'ra': [], 'dec': []}

        def fill_nan(diction):
            diction['logA'].append(np.NaN)
            diction['ind'].append(np.NaN)
            diction['logB'].append(np.NaN)
            diction['logC'].append(np.NaN)
            diction['logA_std'].append(np.NaN)
            diction['ind_std'].append(np.NaN)
            diction['logB_std'].append(np.NaN)
            diction['logC_std'].append(np.NaN)

        # Define an apodization kernel
        # for y, x in ProgressBar(posns):
        for y, x in posns:
            if y == 0 or x == 0:
                fill_nan(fit_results)
                continue
            if y > mom0.shape[0] - cutout_size or y > mom0.shape[0] - cutout_size:
                fill_nan(fit_results)
                continue

            cutout_slice = (slice(max(y - cutout_size, 0), y + cutout_size),
                            slice(max(x - cutout_size, 0), x + cutout_size))

            mom0_cutout = mom0[cutout_slice].copy()

            if (np.isnan(mom0_cutout).sum() / float(mom0_cutout.size)) > 0.1:
                fill_nan(fit_results)
                continue

            # Check to see if the pspec file already exists:
            save_pspec_name = osjoin(pspec_posn_folder,
                                     f"{fitinfo_dict[gal]['filename'].rstrip('fits')}_y_{y}_x_{x}_pspec.pkl")

            if not os.path.exists(save_pspec_name):

                pspec = PowerSpectrum(mom0_cutout.hdu, beam=mom0.beam,
                                      distance=fitinfo_dict[gal]['distance'])
                pspec.compute_pspec(use_pyfftw=False, threads=ncores,
                                    apodize_kernel=fitinfo_dict[gal]['apod_kern'],
                                    alpha=alpha)
                pspec.compute_radial_pspec()

                pspec.save_results(save_pspec_name)

            else:

                pspec = PowerSpectrum.load_results(save_pspec_name)
                pspec.load_beam()

            # pspec.run(high_cut=10**-1.29 / u.pix, low_cut=10**-2. / u.pix,
            #           fit_2D=True,
            #           verbose=False,
            #           # save_name=os.path.join(perposn_figfolder, "{}.png".format(out_name)),
            #           beam_correct=True,
            #           # xunit=u.pc**-1,
            #           apodize_kernel='tukey', alpha=alpha,
            #           radial_pspec_kwargs={'binsize': 2.0},
            #           fit_kwargs={'weighted_fit': True})

            # pspec.save_results(os.path.join(perposn_figfolder,
            #                                 "{}.pkl".format(out_name)))

            # plt.draw()
            # raw_input("{0}--{1}".format(y, x))
            # plt.clf()


            beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
            beam_size = beam_size.value
            beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

            if fitinfo_dict[gal]['high_cut'] is not None:
                high_cut = fitinfo_dict[gal]['high_cut']
            else:
                # high_cut = (1 / (beam_gauss_width * 1.5))
                high_cut = (1 / (beam_gauss_width * 5.))
                # high_cut = (1 / (beam_gauss_width))

            fit_mask = pspec.freqs.value < high_cut

            # And cut out the largest scales due to expected deviations with
            # small stddev
            # fit_mask[:2] = False

            freqs = pspec.freqs.value[fit_mask]
            ps1D = pspec.ps1D[fit_mask]
            ps1D_stddev = pspec.ps1D_stddev[fit_mask]

            def beam_model(f):
                return gaussian_beam(f, beam_gauss_width)

            fixB = True
            # fixB = False

            # noise_term = True
            noise_term = False

            out, summ, trace, fit_model = \
                fit_pspec_model(freqs, ps1D,
                                ps1D_stddev,
                                beam_model=beam_model,
                                nsamp=nsamp,
                                fixB=fixB,
                                noise_term=noise_term,
                                progressbar=False,
                                cores=ncores)

            # row_names.append(gal.lower())

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

            # plt.draw()
            # raw_input("{0} {1}".format(y, x))
            # plt.clf()

            plt.figure(figsize=(8.4, 2.9))

            plt.subplot(121)
            # plt.title("Fit_params: {}".format(out[0]))
            plt.loglog(pspec.freqs.value, pspec.ps1D, 'k', zorder=-10)

            # if noise_term:
            #     beam_amp = 10**(max(fit_results['logA'][-1],
            #                         fit_results['logB'][-1],
            #                         fit_results['logC'][-1]) - 1.)
            beam_amp = 10**(max(fit_results['logA'][-1],
                                fit_results['logB'][-1]) - 1.)

            logA = fit_results['logA'][-1]
            ind = fit_results['ind'][-1]
            mean_pars = [logA, ind]

            if fixB:
                mean_pars.append(-20)
            else:
                logB = fit_results['logB'][-1]
                mean_pars.append(logB)

            if noise_term:
                logC = fit_results['logC'][-1]
                mean_pars.append(logC)

            plt.loglog(freqs, fit_model(freqs, *mean_pars), 'r--',
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
                pars = [logA, ind]

                if fixB:
                    pars.append(-20)
                else:
                    logB = trace.get_values('logB')[rint]
                    pars.append(logB)

                if noise_term:
                    logC = trace.get_values('logC')[rint]
                    pars.append(logC)

                plt.loglog(freqs, fit_model(freqs, *pars),
                           color='gray', alpha=0.2,
                           linewidth=3, zorder=-1)

            plt.axvline(1 / beam_size, linestyle=':', linewidth=4,
                        alpha=0.8, color='gray')

            plt.grid()

            plt.subplot(122)
            plt.imshow(np.log10(pspec.ps2D), origin='lower', cmap='plasma')
            cbar = plt.colorbar()

            # Add contour showing region fit
            yy_freq, xx_freq = make_radial_freq_arrays(pspec.ps2D.shape)

            freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)

            mask = freqs_dist <= high_cut

            plt.contour(mask, colors=['k'], linestyles=['--'])

            plt.tight_layout()

            plt.draw()

            base_name = os.path.basename(filename)

            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.pspec_wbeam.png".format(base_name.rstrip(".fits"), y, x))

            print(plot_savename)
            print("Fit_params: {}".format(out[0]))
            # print("Fit_errs: {}".format(np.sqrt(np.abs(np.diag(out[1])))))
            print("Fit_errs: {}".format(out[1]))
            if make_interactive:
                input("?")

            plt.savefig(plot_savename)

            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.pspec_wbeam.pdf".format(base_name.rstrip(".fits"), y, x))
            plt.savefig(plot_savename)

            plt.close()

            tr_plot = pm.traceplot(trace)

            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.pspec_wbeam_traceplot.png".format(base_name.rstrip(".fits"), y, x))

            plt.draw()
            if make_interactive:
                input("?")

            plt.savefig(plot_savename)

            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.pspec_wbeam_traceplot.pdf".format(base_name.rstrip(".fits"), y, x))
            plt.savefig(plot_savename)

            plt.close()

            # OneD spectrum by itself

            plt.figure(figsize=(4.2, 2.9))

            phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

            phys_scales = 1 / phys_freqs

            plt.loglog(phys_scales, pspec.ps1D, 'k', zorder=-10)

            beam_amp = 10**(max(fit_results['logA'][-1],
                                fit_results['logB'][-1]) - 1.)

            plt.loglog(phys_scales[fit_mask],
                       fit_model(freqs, *mean_pars), 'r--',
                       linewidth=3, label='Fit')
            plt.loglog(phys_scales[fit_mask],
                       beam_amp * beam_model(freqs), 'r:', label='PSF')

            plt.legend(frameon=True, loc='upper right')

            # Also plot a set of 10 random parameter draws

            # Get some random draws
            randints = np.random.randint(0, high=nsamp, size=10)

            for rint in randints:
                logA = trace.get_values('logA')[rint]
                ind = trace.get_values('index')[rint]
                pars = [logA, ind]

                if fixB:
                    pars.append(-20)
                else:
                    logB = trace.get_values('logB')[rint]
                    pars.append(logB)

                if noise_term:
                    logC = trace.get_values('logC')[rint]
                    pars.append(logC)

                plt.loglog(phys_scales[fit_mask], fit_model(freqs, *pars),
                           color='gray', alpha=0.2,
                           linewidth=3, zorder=-1)

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

            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.1Dpspec_wbeam.png".format(base_name.rstrip(".fits"), y, x))
            plt.savefig(plot_savename)
            plot_savename = osjoin(perposn_figfolder, "{0}_{1}_{2}.1Dpspec_wbeam.pdf".format(base_name.rstrip(".fits"), y, x))
            plt.savefig(plot_savename)

            plt.close()

        fit_results['x'] = posns[:, 1]
        fit_results['y'] = posns[:, 0]

        # Add in RA and Dec of the centre points
        dec, ra = proj_coldens.spatial_coordinate_map

        for y, x in zip(posns[:, 0], posns[:, 1]):
            fit_results['ra'].append(ra[y, x].value)
            fit_results['dec'].append(dec[y, x].value)

        df = pd.DataFrame(fit_results)  # , index=row_names)
        df.to_csv(osjoin(data_path, "{0}_pspec_perpoint_coldens_fit_results_regionsize_{1}pix.csv".format(gal, cutout_size)))
