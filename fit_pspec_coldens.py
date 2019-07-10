
'''
Fit the column density power-spectra for the 4 galaxies
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

make_interactive = False

if not plt.isinteractive() and make_interactive:
    plt.ion()

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


do_makepspec = False
do_fitpspec = True

ncores = 6

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
# data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

fitinfo_dict = dict()

fitinfo_dict["LMC"] = {'filename': r"lmc_dust.surface.density_FB.beta=1.8_gauss53.4_regrid_mwsub.fits",
                       'beam': Beam(53.4 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 50.1 * u.kpc}
fitinfo_dict["SMC"] = {'filename': r"smc_dust.surface.density_FB.beta=1.8_gauss43.2_regrid_mwsub.fits",
                       'beam': Beam(43.2 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 62.1 * u.kpc}
fitinfo_dict["M33"] = {'filename': r"m33_dust.surface.density_FB.beta=1.8_gauss41.0_regrid_bksub.fits",
                       'beam': Beam(41.0 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 840 * u.kpc}
fitinfo_dict["M31"] = {'filename': r"m31_dust.surface.density_FB.beta=1.8_gauss46.3_regrid_bksub.fits",
                       'beam': Beam(46.3 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 744 * u.kpc}

if do_makepspec:

    skip_check = False

    for gal in fitinfo_dict:

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

        # Look at the uncertainty map
        masked_errs = hdu_coldens[0].data[2].squeeze()
        masked_errs[~coldens_mask] = np.NaN
        proj_coldens_err = Projection.from_hdu(fits.PrimaryHDU(masked_errs,
                                                               hdu_coldens[0].header))

        proj_coldens_err = proj_coldens_err[nd.find_objects(coldens_mask)[0]]

        proj_coldens_err = proj_coldens_err.with_beam(fitinfo_dict[gal]['beam'])

        save_name = "{0}_coldens.pspec.pkl".format(gal.lower())

        if os.path.exists(osjoin(data_path, gal, save_name)) and skip_check:
            print("Already saved pspec for {}. Skipping".format(filename))
            continue
        else:
            os.system("rm {}".format(osjoin(data_path, gal, save_name)))

        # We also want to account for the shape of the masked data
        # (mostly for M31)
        # norm_mask = np.isfinite(proj_coldens).astype(np.float) / \
        #     np.isfinite(proj_coldens).sum()
        # pspec_mask = PowerSpectrum(fits.PrimaryHDU(norm_mask, proj_coldens.header),
        #                            distance=fitinfo_dict[gal]['distance'])
        # pspec_mask.run(verbose=False, beam_correct=False, fit_2D=False,
        #                high_cut=0.1 / u.pix,
        #                use_pyfftw=True, threads=ncores,
        #                apodize_kernel=fitinfo_dict[gal]['apod_kern'])

        pspec = PowerSpectrum(proj_coldens,
                              distance=fitinfo_dict[gal]['distance'])
        pspec.compute_pspec(use_pyfftw=True, threads=ncores,
                            apodize_kernel=fitinfo_dict[gal]['apod_kern'])
        # Divide out the normalized mask pspec
        # pspec._ps2D /= pspec_mask.ps2D
        pspec.compute_radial_pspec()
        # pspec.fit_pspec()  # high_cut=0.1 / u.pix,)
        # pspec.run(verbose=False, beam_correct=False, fit_2D=False,
        #           high_cut=0.1 / u.pix,
        #           use_pyfftw=True, threads=ncores,
        #           apodize_kernel=fitinfo_dict[gal]['apod_kern'])

        if make_interactive:
            print(pspec.slope)
            pspec.plot_fit(show_residual=False, show_2D=True)
            plt.draw()
            input(gal)
            plt.close()

        full_save_name = osjoin(data_path, gal, save_name)
        pspec.save_results(full_save_name,
                           keep_data=False)

if do_fitpspec:

    nsamp = 6000

    row_names = []

    fit_results = {'logA': [], 'ind': [], 'logB': [], 'logC': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': [],
                   'logC_std': []}

    # Load model functions
    repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
    code_name = os.path.join(repo_path, "models.py")
    exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

    for gal in fitinfo_dict:

        # Make a plot output folder
        plot_folder = osjoin(data_path, "{}_plots".format(gal))
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        filename = "{0}_coldens.pspec.pkl".format(gal.lower())

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

        if fitinfo_dict[gal]['high_cut'] is not None:
            high_cut = fitinfo_dict[gal]['high_cut']
        else:
            high_cut = (1 / (beam_gauss_width * 1.5))
            # high_cut = (1 / (beam_gauss_width * 5.))
            # high_cut = (1 / (beam_gauss_width))

        fit_mask = pspec.freqs.value < high_cut

        # And cut out the largest scales due to expected deviations with
        # small stddev
        fit_mask[:2] = False

        freqs = pspec.freqs.value[fit_mask]
        ps1D = pspec.ps1D[fit_mask]
        ps1D_stddev = pspec.ps1D_stddev[fit_mask]

        def beam_model(f):
            return gaussian_beam(f, beam_gauss_width)

        # fixB = True
        fixB = False

        noise_term = True

        out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                      ps1D_stddev,
                                                      beam_model=beam_model,
                                                      nsamp=nsamp,
                                                      fixB=fixB,
                                                      noise_term=noise_term)

        row_names.append(gal.lower())

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

        # Hang onto the random samples for the paper plots.
        rand_pars = []

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

            rand_pars.append(pars)

            plt.loglog(freqs, fit_model(freqs, *pars),
                       color='gray', alpha=0.2,
                       linewidth=3, zorder=-1)

        # Save the random samples to a npy file
        randfilename = osjoin(data_path, gal.upper(),
                              f"{filename}_param_samples.npy")
        np.save(randfilename, np.array(rand_pars))

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

        plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.png".format(filename.rstrip(".fits")))
        plt.savefig(plot_savename)
        plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.pdf".format(filename.rstrip(".fits")))
        plt.savefig(plot_savename)

        plt.close()

    df = pd.DataFrame(fit_results, index=row_names)
    df.to_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"))
