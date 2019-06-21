
'''
Fit power spectra to individual band images.
'''

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.optimize import curve_fit
from radio_beam import Beam
import pymc3 as pm
import pandas as pd

make_interactive = False

if not plt.isinteractive() and make_interactive:
    plt.ion()
else:
    plt.ioff()

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


# Load model functions
repo_path = os.path.expanduser("~/ownCloud/code_development/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
from fit_settings import fitinfo_dict

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
# Running on tycho
# data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")


# Elements of the dictionary are:
# Band name: eff Gaussian width at original res., low freq cut for fit,
# high freq cut for fit, low intensity cut to mask in image, high intensity
# cut to mask in image, use beam shape in fit

# A freq cut-off of None defaults to 3 times the Gaussian beam

gals = ['LMC', 'SMC', 'M33', 'M31']

# Run at original and moderate convolution to Gaussian

res_types = ['orig', 'mod']

distances = [50.1 * u.kpc, 62.1 * u.kpc, 840 * u.kpc, 744 * u.kpc]

skip_check = False

fit_results = {'logA': [], 'ind': [], 'logB': [],
               'logA_std': [], 'ind_std': [], 'logB_std': []}
row_names = []

for gal, dist in zip(gals, distances):

    print("On {}".format(gal))

    # Make a plot output folder
    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    for name in fitinfo_dict[gal]:

        print("On {}".format(name))

        for res_type in res_types:

            print("Resolution {}".format(res_type))

            if res_type == 'orig':
                filename = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)
            else:
                filename = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(), name, res_type)

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

                kern_fpath = osjoin(data_path, gal, kern_save_name)
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

            out, summ, trace, fit_model = fit_pspec_model(freqs, ps1D,
                                                          ps1D_stddev,
                                                          beam_model=fit_beam_model,
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

        # plt.draw()
        # print(out[0])
        # input(filename)
        # plt.clf()

df = pd.DataFrame(fit_results, index=row_names)
df.to_csv(os.path.join(data_path, "pspec_fit_results.csv"))
