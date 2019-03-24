
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
from scipy.optimize import curve_fit
from functools import partial
from radio_beam import Beam

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

names = {'mips160': Beam(38.8 * u.arcsec),
         'mips24': Beam(6.5 * u.arcsec),
         'mips70': Beam(18.7 * u.arcsec),
         'pacs100': Beam(7.1 * u.arcsec),
         'pacs160': Beam(11.2 * u.arcsec),
         'pacs70': Beam(5.8 * u.arcsec),
         'spire250': Beam(18.2 * u.arcsec),
         'spire350': Beam(25 * u.arcsec),
         'spire500': Beam(36.4 * u.arcsec)}

gals = ['LMC', 'SMC', 'M33', 'M31']

distances = [50.1 * u.kpc, 62.1 * u.kpc, 840 * u.kpc, 744 * u.kpc]

# Some images are large. Run fft in parallel
ncores = 6

for gal, dist in zip(gals, distances):

    print("On {}".format(gal))

    for name in names:

        print("On {}".format(name))

        filename = "{0}_{1}_mjysr.fits".format(gal.lower(), name)

        if not os.path.exists(osjoin(data_path, gal, filename)):
            print("Could not find {}. Skipping".format(filename))
            continue

        save_name = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)

        # For now skip already saved power-spectra
        if os.path.exists(osjoin(data_path, gal, save_name)):
            print("Already saved pspec for {}. Skipping".format(filename))
            continue

        hdu = fits.open(osjoin(data_path, gal, filename))
        proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                   hdu[0].header))
        # Attach equiv Gaussian beam
        proj = proj.with_beam(names[name])

        # Take minimal shape. Remove empty space.
        proj = proj[nd.find_objects(np.isfinite(proj))[0]]

        pspec = PowerSpectrum(proj, distance=dist)
        pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                  high_cut=0.1 / u.pix,
                  use_pyfftw=True, threads=ncores)

        pspec.save_results(osjoin(data_path, gal, save_name), keep_data=False)

        del pspec, proj, hdu
