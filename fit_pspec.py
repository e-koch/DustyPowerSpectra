
'''
Fit power spectra to individual band images.
'''

import os
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
         # 'pacs70': Beam(5.8 * u.arcsec),
         'spire250': Beam(18.2 * u.arcsec),
         'spire350': Beam(25 * u.arcsec),
         'spire500': Beam(36.4 * u.arcsec)}

gals = ['LMC', 'SMC', 'M33', 'M31']

# dist_cuts = [5, 3, 12, 20] * u.kpc

distances = [50.1 * u.kpc, 62.1 * u.kpc, 840 * u.kpc, 744 * u.kpc]

# Some images are large. Run fft in parallel
ncores = 6

img_view = False

skip_check = False

for gal, dist in zip(gals, distances):

    print("On {}".format(gal))

    # Make a plot output folder
    plot_folder = osjoin(data_path, "{}_plots".format(gal))
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    for name in names:

        print("On {}".format(name))

        filename = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)

        # For the convolved maps, the scale changes so use glob
        # filename = "{0}_{1}_gauss*.fits".format(gal.lower(), name)
        # matches = glob(osjoin(data_path, gal, filename))
        # if len(matches) == 0:
        #     raise ValueError("Problem")
        # filename = matches[1]

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

        # Fit on scales > 3 pixels to avoid flattening from pixelization
        # fit_mask = pspec.freqs.value < 1 / 3.
        fit_mask = pspec.freqs.value < 1 / (beam_gauss_width * 1.5)

        freqs = pspec.freqs.value[fit_mask]
        ps1D = pspec.ps1D[fit_mask]
        ps1D_stddev = pspec.ps1D_stddev[fit_mask]

        def powerlaw_model_wbeam(f, A, ind, B=0.):
            # The beam component is squared. Otherwise the exponent is 1/2
            return (A * f**-ind + B) * \
                np.exp(-f**2 * np.pi**2 * 4 * beam_gauss_width**2)

        out = curve_fit(powerlaw_model_wbeam,
                        freqs,
                        ps1D,
                        p0=(1e2, 2.2, 1.0),
                        # p0=(1e2, 2.2),  # 1.0),
                        sigma=ps1D_stddev,
                        absolute_sigma=True, maxfev=100000)

        plt.subplot(121)
        plt.title("Fit_params: {}".format(out[0]))
        plt.loglog(pspec.freqs.value, pspec.ps1D)
        plt.loglog(freqs, powerlaw_model_wbeam(freqs, *out[0]))

        plt.subplot(122)
        plt.title(filename)
        plt.imshow(np.log10(pspec.ps2D), origin='lower')

        plt.tight_layout()

        plt.draw()

        plt.savefig(osjoin(plot_folder, "{}.pspec_wbeam.png".format(filename.rstrip(".fits"))))

        plt.close()

        # plt.draw()
        # print(out[0])
        # input(filename)
        # plt.clf()
