
'''
Compute and fit the M33 HI power-spectrum to compare with the dust and IR
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
do_makepspec_conv = False
do_fitpspec_conv = True

ncores = 6

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

hi_name = osjoin(data_path,
                 "M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.moment0_Kkms.fits")

hi_pspec_name = f"{hi_name.rstrip('fits')}pspec.pkl"


hi_pspec_name_conv = f"{hi_name.rstrip('fits')}conv.pspec.pkl"


if do_makepspec:

    hdu = fits.open(hi_name)

    pspec = PowerSpectrum(hdu[0], distance=840 * u.kpc)

    pspec.run(verbose=False, fit_2D=False)

    pspec.save_results(hi_pspec_name)

if do_fitpspec:

    # Fit the same as the dust column density model

    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    nsamp = 6000

    row_names = []

    gal = 'M33'

    fit_results = {'logA': [], 'ind': [], 'logB': [], 'logC': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': [],
                   'logC_std': []}

    # Load model functions
    repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
    code_name = os.path.join(repo_path, "models.py")
    exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

    pspec = PowerSpectrum.load_results(hi_pspec_name)

    filename = hi_pspec_name_conv

    # Beam doesn't stay cached. Don't know why
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

    fixB = False

    noise_term = False

    out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                  ps1D_stddev,
                                                  beam_model=beam_model,
                                                  nsamp=nsamp,
                                                  fixB=fixB,
                                                  noise_term=noise_term)

    row_names.append('m33')

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

    plt.savefig(plot_savename)

    plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam.pdf".format(filename.rstrip(".fits")))
    plt.savefig(plot_savename)

    plt.close()

    tr_plot = pm.traceplot(trace)

    plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam_traceplot.png".format(filename.rstrip(".fits")))

    plt.draw()

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
    df.to_csv(osjoin(data_path, "pspec_hi_m33_fit_results.csv"))


if do_makepspec_conv:

    gal = 'M33'

    hdu = fits.open(hi_name)

    proj = Projection.from_hdu(hdu)

    beam = Beam(41 * u.arcsec)

    proj_conv = proj.convolve_to(beam)

    pspec = PowerSpectrum(proj_conv, distance=840 * u.kpc)

    pspec.run(verbose=False, fit_2D=False)

    pspec.save_results(hi_pspec_name_conv)

if do_fitpspec_conv:

    # Fit the same as the dust column density model

    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    nsamp = 6000

    row_names = []

    gal = 'M33'

    fit_results = {'logA': [], 'ind': [], 'logB': [], 'logC': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': [],
                   'logC_std': []}

    # Load model functions
    repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
    code_name = os.path.join(repo_path, "models.py")
    exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

    pspec = PowerSpectrum.load_results(hi_pspec_name_conv)

    filename = hi_pspec_name

    # Beam doesn't stay cached. Don't know why
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

    fixB = False

    noise_term = False

    out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                  ps1D_stddev,
                                                  beam_model=beam_model,
                                                  nsamp=nsamp,
                                                  fixB=fixB,
                                                  noise_term=noise_term)

    row_names.append('m33')

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

    plt.savefig(plot_savename)

    plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam.pdf".format(filename.rstrip(".fits")))
    plt.savefig(plot_savename)

    plt.close()

    tr_plot = pm.traceplot(trace)

    plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam_traceplot.png".format(filename.rstrip(".fits")))

    plt.draw()

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
    df.to_csv(osjoin(data_path, "pspec_hi_conv_m33_fit_results.csv"))
