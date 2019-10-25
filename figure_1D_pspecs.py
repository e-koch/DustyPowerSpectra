
'''
Make set of figures showing the power-spectra for the 4 galaxies at a common
physical scale.

Make one figure per band.
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
from matplotlib.ticker import MaxNLocator

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


repo_path = os.path.expanduser("~/ownCloud/project_code/DustyPowerSpectra/")
code_name = os.path.join(repo_path, "models.py")
exec(compile(open(code_name, "rb").read(), code_name, 'exec'))

# Load in fit settings
# fitsetting_name = os.path.join(repo_path, "fit_settings.py")
# exec(compile(open(code_name, "rb").read(), fitsetting_name, 'exec'))
from fit_settings import fitinfo_dict

from plotting_styles import twocolumn_figure


twocolumn_figure()

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Deal with individual bands first. Column density plots follow.

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)


df = pd.read_csv(osjoin(data_path, "pspec_fit_results.csv"), index_col=0)

# Broken plaw fit params for MIPS 24
df_bplaw = pd.read_csv(osjoin(data_path, 'model_comparison', "pspec_modecompare_fit_results.csv"), index_col=0)

gals = {'LMC': 50.1 * u.kpc, 'SMC': 62.1 * u.kpc,
        'M31': 744 * u.kpc, 'M33': 840 * u.kpc}

bands = ['MIPS 24', 'MIPS 70', 'PACS 100', 'MIPS 160', 'PACS 160',
         'SPIRE 250', 'SPIRE 350', 'SPIRE 500']

bands = {'mips24': "MIPS 24",
         'mips70': "MIPS 70",
         'pacs100': "PACS 100",
         'mips160': "MIPS 160",
         'pacs160': "PACS 160",
         'spire250': "SPIRE 250",
         'spire350': "SPIRE 350",
         'spire500': "SPIRE 500"}

resolutions = {'orig': "Original", 'mod': "Convolved"}

for band in bands:

    for res_type in resolutions:

        fig, axs = plt.subplots(4, 2, sharex=True, sharey=False,
                                figsize=(8.4, 7.))

        for i, (gal, ax) in enumerate(zip(gals, axs)):

            # Tie y-axes in each row together
            ax[0].get_shared_y_axes().join(ax[0], ax[1])

            # Use broken plaw for MIPS 24 LMC

            if band == 'mips24' and gal == 'LMC':
                fit_params = df_bplaw.loc[f"{gal.lower()}_{band}_{res_type}_mean"]
            else:
                fit_params = df.loc[f"{gal.lower()}_{band}_{res_type}"]

            if res_type == 'orig':
                filename = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), band)
            else:
                filename = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(),
                                                                band, res_type)

            pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))
            pspec.load_beam()

            beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
            beam_size = beam_size.value
            beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

            if fitinfo_dict[gal][band]['high_cut'] is not None:
                high_cut = fitinfo_dict[gal][band]['high_cut']

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
            beam_freqs = pspec.freqs.value[pspec.freqs.value < (1 / (beam_gauss_width * 3.))]
            ps1D = pspec.ps1D[fit_mask]
            ps1D_stddev = pspec.ps1D_stddev[fit_mask]

            phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

            phys_scales = 1 / phys_freqs

            # One side only shows the power-spectrum
            ax[0].plot(np.log10(phys_scales), np.log10(pspec.ps1D), 'k')

            yerr = 0.434 * pspec.ps1D_stddev / pspec.ps1D
            ax[0].fill_between(np.log10(phys_scales),
                               np.log10(pspec.ps1D) - yerr,
                               y2=np.log10(pspec.ps1D) + yerr,
                               color='k',
                               alpha=0.3)

            # And the beam
            phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

            ax[0].axvline(np.log10(1 / phys_beam), linestyle=':', linewidth=4,
                          alpha=0.8, color='gray')

            # And the PSF shape
            if res_type == 'orig':
                kern_save_name = f"{band}_kernel_{gal.lower()}.pspec.pkl"

                kern_fpath = osjoin(data_path, gal, kern_save_name)
                if not os.path.exists(kern_fpath):
                    raise OSError("Pspec {0} not found.".format(kern_fpath))

                beam_model = make_psf_beam_function(kern_fpath)

            # otherwise use a gaussian beam model
            else:

                def beam_model(f):
                    return gaussian_beam(f, beam_gauss_width)

            try:
                beam_amp = 10**(max(fit_params.logA, fit_params.logB) - 0.75)
            except AttributeError:
                beam_amp = 10**(max(fit_params.logA_plaw,
                                    fit_params.logB_plaw) - 0.75)


            ax[0].plot(np.log10(phys_scales[pspec.freqs.value < (1 / (beam_gauss_width * 3.))]),
                       np.log10(beam_amp * beam_model(beam_freqs)),
                       'r-.',
                       linewidth=2)

            # Invert once b/c all x-axes are the same
            if i == 0:
                ax[0].invert_xaxis()
            ax[0].grid()

            # Galaxy label
            ax[0].text(0.95, 0.85, gal, size=12,
                       bbox={"boxstyle": "round", "facecolor": "w",
                             'edgecolor': 'gray'},
                       transform=ax[0].transAxes,
                       verticalalignment='top',
                       horizontalalignment='right')

            # Give band and resolution
            if i == 0:
                ax[0].text(0.05, 0.15,
                           f"{bands[band]}\n{resolutions[res_type]}",
                           size=12,
                           bbox={"boxstyle": "round", "facecolor": "w",
                                 'edgecolor': 'gray'},
                           transform=ax[0].transAxes,
                           verticalalignment='bottom',
                           horizontalalignment='left')

            ymin = min(np.log10(beam_amp * beam_model(np.array([1 / (1.5 * beam_size)]))),
                       np.log10(pspec.ps1D[np.argmin(np.abs(freqs - 1 / (1.5 * beam_size)))]))

            ymax = np.max(np.log10(pspec.ps1D) + yerr)

            yrange = ymax - ymin

            ax[0].set_ylim(ymin - 0.05 * yrange,
                           ymax + 0.05 * yrange)

            ax[0].yaxis.set_major_locator(MaxNLocator(5, integer=True))

            # Second plot shows the power-spectrum with the model.

            # Check if the fit uses the PSF
            if fitinfo_dict[gal][band]['use_beam']:
                # One case of the broken plaw model
                if gal == 'LMC' and band == 'mips24':
                    fit_model = lambda f, args: broken_powerlaw_model(f, *args) * beam_model(f)
                else:
                    fit_model = lambda f, args: powerlaw_model(f, *args) * beam_model(f)
            else:
                fit_model = lambda f, args: powerlaw_model(f, *args)

            ax[1].plot(np.log10(phys_scales), np.log10(pspec.ps1D),
                       'k', zorder=-10)

            if band == 'mips24' and gal == 'LMC':
                # Broken plaw
                print(fit_params)
                ax[1].plot(np.log10(phys_scales[fit_mask]),
                           np.log10(fit_model(freqs, [fit_params.logA_brokplaw,
                                                      fit_params.index1_brokplaw,
                                                      fit_params.index1_brokplaw + fit_params.index2_brokplaw,
                                                      fit_params.break_f_brokplaw,
                                                      ])),
                           'r--',
                           linewidth=3, label='Fit')

                # Also indicate the break point
                phys_break = pspec._spatial_freq_unit_conversion(fit_params.break_f_brokplaw / u.pix, u.pc**-1).value
                ax[1].axvline(np.log10(1 / phys_break),
                              linestyle=':', linewidth=3, alpha=0.7,
                              color='gray')

                ax[1].text(np.log10(1 / phys_break), 11,
                           f"Break at\n{int(round(1 / phys_break))} pc",
                           color='gray',
                           rotation=90, horizontalalignment='right',
                           verticalalignment='center',
                           size=12)

            else:

                ax[1].plot(np.log10(phys_scales[fit_mask]),
                           np.log10(fit_model(freqs, [fit_params.logA,
                                                      fit_params.ind,
                                                      fit_params.logB])),
                           'r--',
                           linewidth=3, label='Fit')

            ax[1].plot(np.log10(phys_scales[pspec.freqs.value < (1 / (beam_gauss_width * 3.))]),
                       np.log10(beam_amp * beam_model(beam_freqs)),
                       'r-.', label='PSF',
                       linewidth=2)

            # Just one legend
            if i == 0:
                ax[1].legend(frameon=True, loc='center left')

            ax[1].axvline(np.log10(1 / phys_beam),
                          linestyle=':', linewidth=4,
                          alpha=0.8, color='gray')

            # Also plot a set of 10 random parameter draws

            # Get some random draws saved from the fits
            if band == 'mips24' and gal == 'LMC':
                # Load existing trace with model
                randfilename = osjoin(data_path, 'model_comparison',
                                      f"{filename}_param_samples.npy")

                rand_pars = np.load(randfilename)

                # print(argh)

                for pars in rand_pars:

                    ax[1].plot(np.log10(phys_scales[fit_mask]),
                               np.log10(fit_model(freqs,
                                                  [pars[0],
                                                   pars[1],
                                                   pars[1] + pars[2],
                                                   pars[3]])),
                               color='gray', alpha=0.25,
                               linewidth=3, zorder=-1)

            else:

                randfilename = osjoin(data_path, gal.upper(),
                                      f"{filename}_param_samples.npy")

                rand_pars = np.load(randfilename)
                for pars in rand_pars:

                    ax[1].plot(np.log10(phys_scales[fit_mask]),
                               np.log10(fit_model(freqs, pars)),
                               color='gray', alpha=0.25,
                               linewidth=3, zorder=-1)

            ax[1].grid()

            # Don't need y ticks here:
            ax[1].set_yticks(ax[0].get_yticks())
            ax[1].set_yticklabels([])
            ax[1].set_ylim(*ax[0].get_ylim())

        axs[-1][0].xaxis.set_major_locator(MaxNLocator(5, integer=True))

        # Force draw to get the tick locations set
        fig.canvas.draw()

        for i, ax in enumerate(axs):
            ax[0].set_yticklabels([r"$10^{{{}}}$".format(ylab.get_text().strip("$"))
                                   for ylab in ax[0].get_yticklabels()])

        axs[-1][0].set_xticklabels([r"$10^{{{}}}$".format(xlab.get_text().strip("$"))
                                    for xlab in axs[-1][0].get_xticklabels()])

        fig.text(0.53, 0.02, 'Scale (pc)', ha='center', va='center')
        fig.text(0.02, 0.53, 'Power', ha='center', va='center', rotation=90)
        fig.subplots_adjust(left=0.08, bottom=0.08,
                            right=0.98, top=0.98,
                            wspace=0.01, hspace=0.05)

        fig.savefig(osjoin(plot_folder, f"all_1Dpspec_{band}_{res_type}.png"))
        fig.savefig(osjoin(plot_folder, f"all_1Dpspec_{band}_{res_type}.pdf"))

        plt.draw()
        input(f"{band}?")

        plt.close()


# Same thing, but for the column density fits

df = pd.read_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"), index_col=0)

fig, axs = plt.subplots(4, 2, sharex=True, sharey=False, figsize=(8.4, 7.))

for i, (gal, ax) in enumerate(zip(gals, axs)):

    fit_params = df.loc[f"{gal.lower()}"]

    filename = "{0}_coldens.pspec.pkl".format(gal.lower())

    pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))
    pspec.load_beam()

    beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
    beam_size = beam_size.value
    beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

    high_cut = (1 / (beam_gauss_width * 1.5))
    fit_mask = pspec.freqs.value < high_cut

    # And cut out the largest scales due to expected deviations with
    # small stddev
    fit_mask[:2] = False

    freqs = pspec.freqs.value[fit_mask]
    beam_freqs = pspec.freqs.value[fit_mask]
    ps1D = pspec.ps1D[fit_mask]
    ps1D_stddev = pspec.ps1D_stddev[fit_mask]

    phys_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, u.pc**-1).value

    phys_scales = 1 / phys_freqs

    # One side only shows the power-spectrum

    ax[0].plot(np.log10(phys_scales), np.log10(pspec.ps1D), 'k')

    yerr = 0.434 * pspec.ps1D_stddev / pspec.ps1D
    ax[0].fill_between(np.log10(phys_scales),
                       np.log10(pspec.ps1D) - yerr,
                       y2=np.log10(pspec.ps1D) + yerr,
                       color='k',
                       alpha=0.3)

    # And the beam
    phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

    ax[0].axvline(np.log10(1 / phys_beam), linestyle=':', linewidth=4,
                  alpha=0.8, color='gray')

    def beam_model(f):
        return gaussian_beam(f, beam_gauss_width)

    beam_amp = 10**(max(fit_params.logA, fit_params.logB) - 0.75)

    ax[0].plot(np.log10(phys_scales[fit_mask]),
               np.log10(beam_amp * beam_model(beam_freqs)), 'r-.',
               linewidth=2)

    ymin = min(np.log10(beam_amp * beam_model(np.array([1 / beam_size]))),
               np.log10(pspec.ps1D[np.argmin(np.abs(freqs - 1 / beam_size))]))

    ax[0].set_ylim(ymin,
                   np.max(np.log10(pspec.ps1D) + yerr) + 0.5)

    ax[0].yaxis.set_major_locator(MaxNLocator(5, integer=True))

    # Invert once b/c all x-axes are the same
    if i == 0:
        ax[0].invert_xaxis()
    ax[0].grid()

    # Galaxy label
    ax[0].text(0.95, 0.85, gal, size=12,
               bbox={"boxstyle": "round", "facecolor": "w",
                     'edgecolor': 'gray'},
               transform=ax[0].transAxes,
               verticalalignment='top',
               horizontalalignment='right')

    # Give band and resolution
    if i == 0:
        ax[0].text(0.05, 0.15, f"Dust\nSurface\nDensity",
                   size=12,
                   bbox={"boxstyle": "round", "facecolor": "w",
                         'edgecolor': 'gray'},
                   transform=ax[0].transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='left')

    # Second plot shows the power-spectrum with the model.

    # Check if the fit uses the PSF
    fit_model = lambda f, args: powerlaw_model(f, *args[:-1]) * beam_model(f) + 10**args[-1]

    ax[1].plot(np.log10(phys_scales), np.log10(pspec.ps1D), 'k', zorder=-10)

    ax[1].plot(np.log10(phys_scales[fit_mask]),
               np.log10(fit_model(freqs, [fit_params.logA, fit_params.ind,
                                          fit_params.logB, fit_params.logC])),
               'r--',
               linewidth=3, label='Fit')
    ax[1].plot(np.log10(phys_scales[fit_mask]),
               np.log10(beam_amp * beam_model(beam_freqs)), 'r-.', label='PSF',
               linewidth=2)

    # Just one legend
    if i == 0:
        ax[1].legend(frameon=True, loc='center left')

    ax[1].axvline(np.log10(1 / phys_beam), linestyle=':', linewidth=4,
                  alpha=0.8, color='gray')

    # Also plot a set of 10 random parameter draws

    # Get some random draws saved from the fits
    randfilename = osjoin(data_path, gal.upper(),
                          f"{gal.lower()}_coldens.pspec.pkl_param_samples.npy")

    rand_pars = np.load(randfilename)
    for pars in rand_pars:

        ax[1].plot(np.log10(phys_scales[fit_mask]),
                   np.log10(fit_model(freqs, pars)),
                   color='gray', alpha=0.25,
                   linewidth=3, zorder=-1)

    ax[1].axvline(np.log10(1 / phys_beam), linestyle=':', linewidth=4,
                  alpha=0.8, color='gray')

    ax[1].grid()

    # Don't need y ticks here:
    ax[1].set_yticklabels([])
    ax[1].set_ylim(*ax[0].get_ylim())


axs[-1][0].xaxis.set_major_locator(MaxNLocator(4, integer=True))

# Force draw to get the tick locations set
fig.canvas.draw()

for ax in axs:
    ax[0].set_yticklabels([r"$10^{{{}}}$".format(ylab.get_text().strip("$"))
                           for ylab in ax[0].get_yticklabels()])


fig.canvas.draw()

# Having trouble doing this in scripted environment. Do it interactively.
# input("Go to script and past in tick label change.")

axs[-1][0].set_xticklabels([r"$10^{{{}}}$".format(xlab.get_text().strip("$"))
                            for xlab in axs[-1][0].get_xticklabels()])


fig.text(0.53, 0.02, 'Scale (pc)', ha='center', va='center')
fig.text(0.02, 0.53, 'Power', ha='center', va='center', rotation=90)
fig.subplots_adjust(left=0.08, bottom=0.08,
                    right=0.98, top=0.98,
                    wspace=0.01, hspace=0.05)

fig.savefig(osjoin(plot_folder, f"all_1Dpspec_coldens.png"))
fig.savefig(osjoin(plot_folder, f"all_1Dpspec_coldens.pdf"))

# plt.draw()
# input(f"{band}?")
plt.close()
