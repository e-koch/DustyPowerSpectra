
'''
Create pixel-scale-matched PSFs and their power-spectra.
'''

import os
from glob import glob
import numpy as np
from astropy.io import fits
from spectral_cube import Projection
from radio_beam import Beam
import astropy.units as u
from astropy.convolution import convolve_fft
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt

from photutils import resize_psf, CosineBellWindow, create_matching_kernel

from turbustat.statistics import PowerSpectrum

if not plt.isinteractive():
    plt.ion()

osjoin = os.path.join


# Running on SegFault w/ data on bigdata
# data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# kern_path = os.path.expanduser("~/bigdata/ekoch/Aniano_kernels/")

data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

kern_path = os.path.expanduser("~/tycho/Aniano_kernels/")


names = {'mips24': ["PSF_MIPS_24.fits", 0],
         'mips70': ["PSF_MIPS_70.fits", 0],
         'pacs100': ["PSF_PACS_100.fits", 0],
         'mips160': ["PSF_MIPS_160.fits", 0],
         'pacs160': ["PSF_PACS_160.fits", 0],
         # 'pacs70': , "mod": } ,
         'spire250': ["PSF_SPIRE_250.fits", 1],
         'spire350': ["PSF_SPIRE_350.fits", 1],
         'spire500': ["PSF_SPIRE_500.fits", 1]}

# gals = ['LMC', 'SMC', 'M33', 'M31']
gals = ['SMC', 'M33', 'M31']

# Some images are large. Run fft in parallel
ncores = 6

img_view = False

skip_check = False

for gal in gals:

    print("On {}".format(gal))

    for name in names:

        print("On {}".format(name))

        filename = "{0}_{1}_mjysr.fits".format(gal.lower(), name)

        if not os.path.exists(osjoin(data_path, gal, filename)):
            print("Could not find {}. Skipping".format(filename))
            continue

        hdu = fits.open(osjoin(data_path, gal, filename))
        proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                   hdu[0].header))

        out_filename = "{0}_{1}_mjysr.fits"\
            .format(gal.lower(), name)

        # Now open the kernel file
        kernfits_name = names[name][0]
        kernfits_ext = names[name][1]

        kernel_filename = osjoin(kern_path, kernfits_name)

        kern_proj = Projection.from_hdu(fits.open(osjoin(kern_path, kernel_filename))[kernfits_ext])

        img_scale = np.abs(proj_plane_pixel_scales(proj.wcs))[0]
        kern_scale = np.abs(proj_plane_pixel_scales(kern_proj.wcs))[0]

        kernel = resize_psf(kern_proj.value, kern_scale, img_scale)

        # Normalize to make a kernel
        kernel /= kernel.sum()

        kern_pspec = PowerSpectrum((kernel, kern_proj.header))
        kern_pspec.run(verbose=False, fit_2D=False)

        save_name = "{0}_kernel_{1}.pspec.pkl".format(name, gal.lower())

        kern_pspec.save_results(osjoin(data_path, gal, save_name),
                                keep_data=True)

        # plt.draw()

        # input("?")

        # # plt.clf()
        # plt.close()
