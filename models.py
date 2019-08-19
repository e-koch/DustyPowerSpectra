
'''
Wrap pymc3 models used as their own functions.
'''

import pymc3 as pm
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from turbustat.statistics import PowerSpectrum


def powerlaw_model(f, logA, ind, logB=-20):
    return 10**logA * f**-ind + 10**logB


def broken_powerlaw_model(f, logA, ind1, ind2, break_f, logB=-20, delta=0.1):

    A = 10**logA
    B = 10**logB

    return A * (f / break_f)**-ind1 * (0.5 * (1 + (f / break_f)**(1 /delta)))**((ind1 - ind2) * delta) + B


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
                    nsamp=6000, step=pm.SMC(parallel=False), cores=1,
                    chains=100,
                    fixB=False, noise_term=False,
                    progressbar=True, return_model=False,):

    def powerlaw_model(f, logA, ind, logB=-20):
        return 10**logA * f**-ind + 10**logB

    if beam_model is not None:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind, logB=-20, logC=-20):
                return powerlaw_model(f, logA, ind, logB) * beam_model(f) + 10**logC
        else:
            def powerlaw_fit_model(f, logA, ind, logB=-20):
                return powerlaw_model(f, logA, ind, logB) * beam_model(f)
    else:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind, logB=-20, logC=-20):
                return powerlaw_model(f, logA, ind, logB) + 10**logC
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

        # Weak Gaussian priors
        # logA = pm.Normal('logA', 0., 10.)
        # ind = pm.Normal('index', 0.0, 3.)
        # if not fixB:
        #     logB = pm.Normal('logB', 0., 10.)
        # else:
        #     logB = -20.

        if noise_term:
            logC = pm.Uniform('logC', -20., 20.)
            # logC = pm.Normal('logC', 0., 10.)
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
                          progressbar=progressbar,
                          cores=cores, chains=chains)

    summ = pm.summary(trace)

    out = [np.array(summ['mean']), np.array(summ['sd'])]

    if return_model:
        return out, summ, trace, powerlaw_fit_model, model
    else:
        return out, summ, trace, powerlaw_fit_model


def fit_broken_pspec_model(freqs, ps1D, ps1D_stddev, beam_model=None,
                           ntune=2000, nsamp=6000,
                           step=pm.SMC(parallel=False), cores=1,
                           chains=100,
                           fixB=False, noise_term=False,
                           progressbar=True, return_model=False,):

    # https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D.html

    def broken_powerlaw_model(f, logA, ind1, ind2, break_f, logB=-20, delta=0.1):

        A = 10**logA
        B = 10**logB

        return A * (f / break_f)**-ind1 * (0.5 * (1 + (f / break_f)**(1 /delta)))**((ind1 - ind2) * delta) + B

    if beam_model is not None:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind1, ind2, break_f, logB=-20, logC=-20):
                return broken_powerlaw_model(f, logA, ind1, ind2, break_f, logB) * beam_model(f) + 10**logC
        else:
            def powerlaw_fit_model(f, logA, ind1, ind2, break_f, logB=-20):
                return broken_powerlaw_model(f, logA, ind1, ind2, break_f, logB) * beam_model(f)
    else:
        if noise_term:
            def powerlaw_fit_model(f, logA, ind1, ind2, break_f, logB=-20, logC=-20):
                return broken_powerlaw_model(f, logA, ind1, ind2, break_f, logB) + 10**logC
        else:
            powerlaw_fit_model = broken_powerlaw_model

    # Try a pymc model to fit

    with pm.Model() as model:

        logA = pm.Uniform('logA', -20., 20.)
        ind1 = pm.Uniform('index1', 0.0, 10.)

        # logA = pm.Normal('logA', 0., 10.)
        # ind1 = pm.Normal('index', 0.0, 3.)
        # if not fixB:
        #     logB = pm.Normal('logB', 0., 10.)
        # else:
        #     logB = -20.

        # Second index is a perturbation on the first.
        ind2 = ind1 + pm.Normal('index2', 0.0, 10.)

        break_f = pm.Uniform('break_f', freqs.min(), freqs.max())

        if not fixB:
            logB = pm.Uniform('logB', -20., 20.)
        else:
            logB = -20.

        if noise_term:
            logC = pm.Uniform('logC', -20., 20.)
            # logC = pm.Normal('logC', 0., 10.)
            ps_vals = pm.Normal('obs',
                                powerlaw_fit_model(freqs, logA, ind1, ind2, break_f, logB=logB, logC=logC),
                                sd=ps1D_stddev,
                                observed=ps1D,
                                shape=freqs.shape)
        else:
            ps_vals = pm.Normal('obs',
                                powerlaw_fit_model(freqs, logA, ind1, ind2, break_f, logB=logB),
                                sd=ps1D_stddev,
                                observed=ps1D,
                                shape=freqs.shape)

        # step = pm.Slice()
        # step = pm.NUTS()
        # step = pm.SMC()

        trace = pm.sample(nsamp, tune=ntune, step=step,
                          progressbar=progressbar,
                          cores=cores, chains=chains)

    summ = pm.summary(trace)

    out = [np.array(summ['mean']), np.array(summ['sd'])]

    if return_model:
        return out, summ, trace, powerlaw_fit_model, model
    else:
        return out, summ, trace, powerlaw_fit_model
