
'''
Fit the LMC images with the maximum slice removing 30 Dor.

Does the bump in the LMC at 24 um go away without 30 Dor?
'''


import os
import numpy as np
from astropy.io import fits
from spectral_cube import Projection
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sb

import pymc3 as pm
import theano.tensor as T
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd

from radio_beam import Beam
from galaxies import Galaxy

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum

do_run_pspec = False
do_fit_pspec = True

# Running on SegFault w/ data on bigdata
# data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

names = {'mips24': Beam(6.5 * u.arcsec)}
name = 'mips24'

gal = 'LMC'
dist = 50.1 * u.kpc

ncores = 6

res_types = ['orig', 'mod']

lmc_mips24_slice = {"30dor": (slice(2620, 9215), slice(218, 6135)),
                    'no30dor': (slice(2160, 8755), slice(4580, 10497))}

skip_check = False

if do_run_pspec:

    for res_type in res_types:

        print("Resolution {}".format(res_type))

        if res_type == 'orig':
            filename = "{0}_{1}_mjysr_cutout.fits".format(gal.lower(), name)
        else:
            filename = "{0}_{1}_{2}_mjysr_cutout.fits".format(gal.lower(), name, res_type)

        if not os.path.exists(osjoin(data_path, gal, filename)):
            print("Could not find {}. Skipping".format(filename))
            continue

        hdu = fits.open(osjoin(data_path, gal, filename))
        proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                   hdu[0].header))
        # Attach equiv Gaussian beam
        # if res_type == 'orig':
        #     proj = proj.with_beam(names[name])

        # With and without 30 Dor
        for slice_name in lmc_mips24_slice:

            slicer = lmc_mips24_slice[slice_name]

            if res_type == 'orig':
                save_name = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(),
                                                                 name, slice_name)
            else:
                save_name = "{0}_{1}_{2}_{3}_mjysr.pspec.pkl".format(gal.lower(),
                                                                     name,
                                                                     res_type,
                                                                     slice_name)

            # For now skip already saved power-spectra
            if os.path.exists(osjoin(data_path, gal, save_name)) and skip_check:
                print("Already saved pspec for {}. Skipping".format(filename))
                continue
            else:
                os.system("rm -f {}".format(osjoin(data_path, gal, save_name)))

            pspec = PowerSpectrum(proj[slicer], distance=dist)
            pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                      high_cut=0.1 / u.pix,
                      use_pyfftw=True, threads=ncores,
                      apodize_kernel='tukey', alpha=0.3)

            pspec.save_results(osjoin(data_path, gal, save_name),
                               keep_data=False)


if do_fit_pspec:

    # Make a plot output folder
    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    fit_results = {'logA': [], 'ind': [], 'logB': [],
                   'logA_std': [], 'ind_std': [], 'logB_std': []}
    row_names = []

    for res_type in res_types:

        print("Resolution {}".format(res_type))

        make_interactive = True

        if make_interactive:
            plt.ion()
        else:
            plt.ioff()

        for slice_name in lmc_mips24_slice:

            if res_type == 'orig':
                filename = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(),
                                                                name, slice_name)
            else:
                filename = "{0}_{1}_{2}_{3}_mjysr.pspec.pkl".format(gal.lower(),
                                                                    name,
                                                                    res_type,
                                                                    slice_name)

            # Load pspec object
            pspec = PowerSpectrum.load_results(osjoin(data_path, gal, filename))

            # Beam doesn't stay cached. Don't know why
            pspec.load_beam()

            beam_size = pspec._beam.major.to(u.deg) / pspec._ang_size.to(u.deg)
            beam_size = beam_size.value
            beam_gauss_width = beam_size / np.sqrt(8 * np.log(2))

            # Fit on scales > 3 pixels to avoid flattening from pixelization
            fit_mask = pspec.freqs.value < (1 / (beam_gauss_width * 3.))

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

                # Load in the pspec and use a spline fit of the pspec for the beam
                # model
                kern_pspec = PowerSpectrum.load_results(kern_fpath)

                largest_val = kern_pspec.ps1D[0]
                smallest_freq = kern_pspec.freqs.value[0]

                spl = InterpolatedUnivariateSpline(kern_pspec.freqs.value,
                                                   kern_pspec.ps1D)

                def beam_model(f):

                    beam_vals = np.empty_like(f)
                    # if on scales larger than the kernel image, return
                    # value on largest scale
                    beam_vals[f < smallest_freq] = largest_val
                    beam_vals[f >= smallest_freq] = spl(f[f >= smallest_freq])

                    return beam_vals

            # otherwise use a gaussian beam model
            else:

                def beam_model(f):
                    # return np.exp(-f**2 * np.pi**2 * 4 * beam_gauss_width**2)
                    return T.exp(-f**2 * np.pi**2 * 4 * beam_gauss_width**2)


            # Model a power-law with a constant modififed by the PSF response
            def powerlaw_model_wbeam(f, logA, ind, logB, pt_on=1.):
                return (10**logA * f**-ind + 10**logB * pt_on) * \
                    beam_model(f)

            # Try a pymc model to fit

            ntune = 2000
            nsamp = 6000

            with pm.Model() as model:

                logA = pm.Uniform('logA', -20., 20.)
                ind = pm.Uniform('index', 0.0, 10.)
                logB = pm.Uniform('logB', -20., 20.)

                ps_vals = pm.Normal('obs',
                                    powerlaw_model_wbeam(freqs, logA, ind, logB),
                                    sd=ps1D_stddev,
                                    observed=ps1D)

                # step = pm.Slice()
                # step2 = pm.Metropolis([logA, ind, logB])
                # step = pm.NUTS()
                step = pm.SMC()

                trace = pm.sample(nsamp, tune=ntune, step=step,
                                  progressbar=True,
                                  cores=ncores)

            summ = pm.summary(trace)

            out = [np.array(summ['mean']), np.array(summ['sd'])]

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

            if res_type == 'orig':
                plt.loglog(freqs, powerlaw_model_wbeam(freqs, *out[0]), 'r--',
                           linewidth=3, label='Fit')
                plt.loglog(freqs,
                           beam_amp * beam_model(freqs), 'r:', label='PSF')
            else:
                plt.loglog(freqs, powerlaw_model_wbeam(freqs, *out[0]).eval(), 'r--',
                           linewidth=3, label='Fit')
                plt.loglog(freqs,
                           beam_amp * beam_model(freqs).eval(), 'r:',
                           label='PSF')

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

                if res_type == 'orig':

                    plt.loglog(freqs, powerlaw_model_wbeam(freqs, *pars),
                               color='gray', alpha=0.7,
                               linewidth=3, zorder=-1)
                else:
                    plt.loglog(freqs,
                               powerlaw_model_wbeam(freqs, *pars).eval(),
                               color='gray', alpha=0.7,
                               linewidth=3, zorder=-1)

            plt.axvline(1 / beam_size, linestyle=':', linewidth=4,
                        alpha=0.8, color='gray')

            plt.grid()

            plt.subplot(122)
            plt.imshow(np.log10(pspec.ps2D), origin='lower')
            plt.colorbar()

            plt.tight_layout()

            plt.draw()

            plot_savename = osjoin(plot_folder, "{0}.pspec_wbeam.png".format(filename.rstrip(".fits")))

            print(plot_savename)
            print("Fit_params: {}".format(out[0]))
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

            if res_type == 'orig':
                plt.loglog(phys_freqs[fit_mask],
                           powerlaw_model_wbeam(freqs, *out[0]), 'r--',
                           linewidth=3, label='Fit')
                plt.loglog(phys_freqs[fit_mask],
                           beam_amp * beam_model(freqs), 'r:', label='PSF')
            else:
                plt.loglog(phys_freqs[fit_mask],
                           powerlaw_model_wbeam(freqs, *out[0]).eval(), 'r--',
                           linewidth=3, label='Fit')
                plt.loglog(phys_freqs[fit_mask],
                           beam_amp * beam_model(freqs).eval(), 'r:',
                           label='PSF')

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

                if res_type == 'orig':

                    plt.loglog(phys_freqs[fit_mask],
                               powerlaw_model_wbeam(freqs, *pars),
                               color='gray', alpha=0.7,
                               linewidth=3, zorder=-1)
                else:
                    plt.loglog(phys_freqs[fit_mask],
                               powerlaw_model_wbeam(freqs, *pars).eval(),
                               color='gray', alpha=0.7,
                               linewidth=3, zorder=-1)

            phys_beam = pspec._spatial_freq_unit_conversion(1 / (beam_size * u.pix), u.pc**-1).value

            plt.axvline(phys_beam, linestyle=':', linewidth=4,
                        alpha=0.8, color='gray')

            plt.grid()

            plt.tight_layout()

            plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.png".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)
            plot_savename = osjoin(plot_folder, "{0}.1Dpspec_wbeam.pdf".format(filename.rstrip(".fits")))
            plt.savefig(plot_savename)

            plt.close()

        df = pd.DataFrame(fit_results, index=row_names)
        df.to_csv(os.path.expanduser("~/tycho/Utomo19_LGdust/pspec_lmc_mip24_30dorcomparison_fit_results.csv"))
