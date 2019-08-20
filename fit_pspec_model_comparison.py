
'''
Compare fitting a single power law vs. a broken power law model.
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

# Make a new directory for the model comparisons
model_comparison_folder = osjoin(data_path, 'model_comparison')
if not os.path.exists(model_comparison_folder):
    os.mkdir(model_comparison_folder)

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

dfs_defined = False

for gal, dist in zip(gals, distances):

    print("On {}".format(gal))

    # Make a plot output folder
    plot_folder = osjoin(model_comparison_folder, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    for name in fitinfo_dict[gal]:

        print("On {}".format(name))

        # Test broken plaw on SPIRE 500
        # if name != 'spire500':
        #     continue

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

            out, summ, trace, fit_model_func, plaw_model = \
                fit_pspec_model(freqs, ps1D,
                                ps1D_stddev,
                                beam_model=fit_beam_model,
                                nsamp=nsamp,
                                fixB=fitinfo_dict[gal][name]['fixB'],
                                return_model=True)

            out_brok, summ_brok, trace_brok, fit_model_func_brok, brok_plaw_model = \
                fit_broken_pspec_model(freqs, ps1D,
                                       ps1D_stddev,
                                       beam_model=fit_beam_model,
                                       nsamp=nsamp,
                                       fixB=fitinfo_dict[gal][name]['fixB'],
                                       return_model=True)

            # Model comparison via WAIC
            plaw_model.name = 'plaw'
            brok_plaw_model.name = 'brok_plaw'

            df_comp_WAIC = pm.compare({plaw_model: trace,
                                       brok_plaw_model: trace_brok},
                                      ic='WAIC')

            plt.figure(figsize=(4.2, 4.2))

            ax = plt.subplot(211)
            # plt.title("Fit_params: {}".format(out[0]))
            ax.loglog(pspec.freqs.value, pspec.ps1D, 'k', zorder=-10)


            beam_amp = 10**(max([summ['mean'].logA, -20 if fitinfo_dict[gal][name]['fixB'] else summ['mean'].logB]) - 1.)

            ax.loglog(freqs,
                      fit_model_func(freqs,
                                     summ['mean']['logA'],
                                     summ['mean']['index'],
                                     logB=-20 if fitinfo_dict[gal][name]['fixB'] else summ['mean'].logB,
                                     ), 'r--',
                      linewidth=3, label='Power-law')

            ax.loglog(freqs,
                      fit_model_func_brok(freqs,
                                          summ_brok['mean']['logA'],
                                          summ_brok['mean']['index1'],
                                          summ_brok['mean']['index2'] + summ_brok['mean']['index1'],
                                          summ_brok['mean']['break_f'],
                                          logB=-20 if fitinfo_dict[gal][name]['fixB'] else summ_brok['mean'].logB,
                                          ),
                      'g-.',
                      linewidth=3, label='Broken power-law')
            ax.loglog(freqs,
                      beam_amp * beam_model(freqs), 'r:', label='PSF')

            ax.set_xlabel("Freq. (1 / pix)")

            ax.legend(frameon=True, loc='upper right')

            ax.axvline(1 / beam_size, linestyle=':', linewidth=4,
                       alpha=0.8, color='gray')

            ax.grid()

            # Model comparison plot
            ax2 = plt.subplot(212)
            pm.compareplot(df_comp_WAIC, ax=ax2)

            plt.tight_layout()

            plot_savename = osjoin(plot_folder, "{0}.pspec_modelcompare.png".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plot_savename = osjoin(plot_folder, "{0}.pspec_modelcompare.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            tr_plot = pm.traceplot(trace_brok)
            plot_savename = osjoin(plot_folder, "{0}.brok_pspec_traceplot.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            tr_plot = pm.traceplot(trace)
            plot_savename = osjoin(plot_folder, "{0}.pspec_traceplot.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

            # Setup to create final save tables.

            row_name = "{0}_{1}_{2}".format(gal.lower(), name, res_type)

            # Rename columns w/ row_name to append into one table
            df_comp_WAIC.index = [f'{row_name}_{ind}' for ind in df_comp_WAIC.index]

            # Combine into one table
            params_df = summ[['mean', 'sd']]
            params_df.index = [f'{ind}_plaw' for ind in params_df.index]
            params_df.columns = [f'{row_name}_{col}' for col in params_df.columns]

            params_df_brok = summ_brok[['mean', 'sd']]
            params_df_brok.index = [f'{ind}_brokplaw' for ind in params_df_brok.index]
            params_df_brok.columns = [f'{row_name}_{col}' for col in params_df_brok.columns]

            all_params_df = params_df.append(params_df_brok)
            # Include the beam size in frequency to reference the break against
            beam_series = pd.Series([1 / beam_gauss_width],
                                    index=[f'{row_name}_mean'],
                                    name='beam_gauss_freq')
            all_params_df = all_params_df.append(beam_series)

            # Append to dfs for all fits
            if not dfs_defined:
                all_fit_params = all_params_df
                all_waic_compare = df_comp_WAIC
                # Append in further iterations.
                dfs_defined = True
            else:
                all_fit_params = all_fit_params.append(all_params_df)
                all_waic_compare = all_waic_compare.append(df_comp_WAIC)

            # Save traces
            # Remove old traces
            trace_save_name = osjoin(model_comparison_folder,
                                     f'{row_name}_plaw_trace')
            if os.path.exists(trace_save_name):
                os.system(f"rm -rf {trace_save_name}")
            pm.save_trace(trace, directory=trace_save_name)

            trace_save_name = osjoin(model_comparison_folder,
                                     f'{row_name}_brokplaw_trace')
            if os.path.exists(trace_save_name):
                os.system(f"rm -rf {trace_save_name}")
            pm.save_trace(trace_brok, directory=trace_save_name)


# Save the fit tables
all_fit_params.to_csv(os.path.join(model_comparison_folder, "pspec_modecompare_fit_results.csv"))
all_waic_compare.to_csv(os.path.join(model_comparison_folder, "pspec_modecompare_waic_results.csv"))
