
'''
Wrap pymc3 models used as their own functions.
'''

import pymc3 as pm
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from turbustat.statistics import PowerSpectrum


def powerlaw_model(f, logA, ind, logB=-20):
    return 10**logA * f**-ind + 10**logB


def gaussian_beam(f, beam_gauss_width):
    return np.exp(-f**2 * np.pi**2 * 4 * beam_gauss_width**2)


def make_psf_beam_function(kern_fpath):
    # Load in the pspec and use a spline fit of the pspec for the beam
    # model
    kern_pspec = PowerSpectrum.load_results(kern_fpath)

    largest_val = kern_pspec.ps1D[0]
    smallest_freq = kern_pspec.freqs.value[0]

    spl = InterpolatedUnivariateSpline(kern_pspec.freqs.value,
                                       kern_pspec.ps1D)

    def beam_model(f):

        beam_vals = np.empty_like(f)
        # beam_vals = T.zeros_like(f)
        # if on scales larger than the kernel image, return
        # value on largest scale
        beam_vals[f < smallest_freq] = largest_val
        beam_vals[f >= smallest_freq] = spl(f[f >= smallest_freq])

        return beam_vals

    return beam_model


def fit_pspec_model(freqs, ps1D, ps1D_stddev, beam_model=None, ntune=2000,
                    nsamp=6000, step=pm.SMC(), fixB=False, noise_term=False):

    def powerlaw_model(f, logA, ind, logB=-20):
        return 10**logA * f**-ind + 10**logB

    if beam_model is not None:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind, logB=-20, logC=-20):
                return powerlaw_model(f, logA, ind, logB) * beam_model(f) + logC
        else:
            def powerlaw_fit_model(f, logA, ind, logB=-20):
                return powerlaw_model(f, logA, ind, logB) * beam_model(f)
    else:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind, logB=-20, logC=-20):
                return powerlaw_model(f, logA, ind, logB) + logC
        else:
            powerlaw_fit_model = powerlaw_model

    # Try a pymc model to fit

    with pm.Model() as model:

        logA = pm.Uniform('logA', -20., 20.)
        ind = pm.Uniform('index', 0.0, 10.)
        if not fixB:
            logB = pm.Uniform('logB', -20., 20.)
        else:
            logB = -20.

        if noise_term:
            logC = pm.Uniform('logC', -20., 20.)
            ps_vals = pm.Normal('obs',
                                powerlaw_fit_model(freqs, logA, ind, logB=logB, logC=logC),
                                sd=ps1D_stddev,
                                observed=ps1D)
        else:
            ps_vals = pm.Normal('obs',
                                powerlaw_fit_model(freqs, logA, ind, logB=logB),
                                sd=ps1D_stddev,
                                observed=ps1D)

        # step = pm.Slice()
        # step = pm.NUTS()
        # step = pm.SMC()

        trace = pm.sample(nsamp, tune=ntune, step=step,
                          progressbar=True,
                          cores=None)

    summ = pm.summary(trace)

    out = [np.array(summ['mean']), np.array(summ['sd'])]

    return out, summ, trace, powerlaw_fit_model
