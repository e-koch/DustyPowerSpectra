
'''
Convolve all maps to the min. resolution with a Gaussian
beam reported in Aniano+2011

Create two versions based on Table 6:
(1) aggressive
(2) moderate

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

from photutils import resize_psf, CosineBellWindow, create_matching_kernel

osjoin = os.path.join


# Running on SegFault w/ data on bigdata
# data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# kern_path = os.path.expanduser("~/bigdata/ekoch/Aniano_kernels/")

data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")

kern_path = os.path.expanduser("~/tycho/Aniano_kernels/")


names = {'mips160': {"agg": "Kernel_HiRes_MIPS_160_to_Gauss_41.fits",
                     "mod": "Kernel_HiRes_MIPS_160_to_Gauss_64.fits"},
         'mips24': {"agg": "Kernel_HiRes_MIPS_24_to_Gauss_08.0.fits",
                    "mod": "Kernel_HiRes_MIPS_24_to_Gauss_11.fits"},
         'mips70': {"agg": "Kernel_HiRes_MIPS_70_to_Gauss_21.fits",
                    "mod": "Kernel_HiRes_MIPS_70_to_Gauss_30.fits"},
         'pacs100': {"agg": "Kernel_HiRes_PACS_100_to_Gauss_07.5.fits",
                     "mod": "Kernel_HiRes_PACS_100_to_Gauss_09.0.fits"},
         'pacs160': {"agg": "Kernel_HiRes_PACS_160_to_Gauss_12.fits",
                     "mod": "Kernel_HiRes_PACS_160_to_Gauss_14.fits"},
         # 'pacs70': {"agg": , "mod": } ,
         'spire250': {"agg": "Kernel_HiRes_SPIRE_250_to_Gauss_20.fits",
                      "mod": "Kernel_HiRes_SPIRE_250_to_Gauss_21.fits"},
         'spire350': {"agg": "Kernel_HiRes_SPIRE_350_to_Gauss_25.fits",
                      "mod": "Kernel_HiRes_SPIRE_350_to_Gauss_28.fits"},
         'spire500': {"agg": "",
                      "mod": "Kernel_HiRes_SPIRE_500_to_Gauss_41.fits"}}
         # Wasn't available for download. But is really close to original

gals = ['LMC', 'SMC', 'M33', 'M31']

# Some images are large. Run fft in parallel
ncores = 6

img_view = False

skip_check = False

for gal in gals:

    print("On {}".format(gal))

    for name in names:

        print("On {}".format(name))

        filename = "{0}_{1}_mjysr.fits".format(gal.lower(), name)

        # For the convolved maps, the scale changes so use glob
        # filename = "{0}_{1}_gauss*.fits".format(gal.lower(), name)
        # matches = glob(osjoin(data_path, gal, filename))
        # if len(matches) == 0:
        #     raise ValueError("Problem")
        # filename = matches[1]

        if not os.path.exists(osjoin(data_path, gal, filename)):
            print("Could not find {}. Skipping".format(filename))
            continue

        hdu = fits.open(osjoin(data_path, gal, filename))
        proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                   hdu[0].header))

        # Loop through 2 convolution settings
        for set_type in ['agg', 'mod']:

            out_filename = "{0}_{1}_{2}_mjysr.fits"\
                .format(gal.lower(), name, set_type)

            # Now open the kernel file
            kernfits_name = names[name][set_type]

            if len(kernfits_name) == 0:
                # Handle the missing SPIRE 500 case
                # Just write out the original data
                proj = proj.with_beam(Beam(36.4 * u.arcsec))
                proj.write(osjoin(data_path, gal, out_filename),
                           overwrite=True)
                continue

            kernel_filename = osjoin(kern_path, kernfits_name)

            kern_proj = Projection.from_hdu(fits.open(osjoin(kern_path, kernel_filename)))

            img_scale = np.abs(proj_plane_pixel_scales(proj.wcs))[0]
            kern_scale = np.abs(proj_plane_pixel_scales(kern_proj.wcs))[0]

            kernel = resize_psf(kern_proj.value, kern_scale, img_scale)

            # Beam sizes are given in the filename
            beam_width = float(kernel_filename.split("Gauss_")[1].rstrip(".fits"))
            beam = Beam(beam_width * u.arcsec)

            # Convolve with the given kernel

            conv_img = convolve_fft(proj.value, kernel, allow_huge=True)

            # Apply original NaN mask
            conv_img[np.isnan(proj)] = np.NaN

            hdu_new = fits.PrimaryHDU(conv_img, proj.header)
            new_proj = Projection.from_hdu(hdu_new)

            new_proj = new_proj.with_beam(beam)

            new_proj.write(osjoin(data_path, gal, out_filename),
                           overwrite=True)

            del new_proj
            del kern_proj

        del proj
