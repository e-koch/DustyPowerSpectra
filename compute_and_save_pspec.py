
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
from galaxies import Galaxy

osjoin = os.path.join

from turbustat.statistics import PowerSpectrum

# Running on SegFault w/ data on bigdata
# data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")
data_path = os.path.expanduser("~/tycho/Utomo19_LGdust/")


fitinfo_dict = dict()

fitinfo_dict["LMC"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'apod_kern': None,
                'low_int_cut': -1000., 'high_int_cut': 4000.},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None}}

fitinfo_dict["SMC"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': 2000.},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                'low_int_cut': -1000., 'high_int_cut': None},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None}}

fitinfo_dict["M33"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None}}

fitinfo_dict["M31"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                'low_int_cut': None, 'high_int_cut': None},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': 'tukey',
                'low_int_cut': None, 'high_int_cut': None}}

gals = ['LMC', 'SMC', 'M33', 'M31']

# Run at original, aggressive convolution to Gaussian, and moderate
# convolution to Gaussian

res_types = ['orig', 'mod']

distances = [50.1 * u.kpc, 62.1 * u.kpc, 840 * u.kpc, 744 * u.kpc]

# Some images are large. Run fft in parallel
ncores = 6

img_view = False

skip_check = True

for gal, dist in zip(gals, distances):

    # Load in the dust column density maps to set the allowed
    # spatial region

    filename_coldens = glob(osjoin(data_path, gal, "*dust.surface.density*.fits"))

    hdu_coldens = fits.open(filename_coldens[0])

    pad_size = 0.5 * u.arcmin

    proj_coldens = Projection.from_hdu(fits.PrimaryHDU(hdu_coldens[0].data[0].squeeze(),
                                                       hdu_coldens[0].header))

    # Get minimal size
    proj_coldens = proj_coldens[nd.find_objects(np.isfinite(proj_coldens))[0]]

    # Get spatial extents.
    # NOTE: extrema for 2D objects broken in spectral-cube! Need to fix...
    lat, lon = proj_coldens.spatial_coordinate_map
    lat_min = lat.min() - pad_size
    lat_max = lat.max() + pad_size
    lon_min = lon.min() - pad_size
    lon_max = lon.max() + pad_size

    def spat_mask_maker(lat_map, lon_map):

        lat_mask = np.logical_and(lat_map > lat_min,
                                  lat_map < lat_max)
        lon_mask = np.logical_and(lon_map > lon_min,
                                  lon_map < lon_max)
        return lat_mask & lon_mask

    print("On {}".format(gal))

    for name in fitinfo_dict[gal]:

        print("On {}".format(name))

        for res_type in res_types:

            print("Resolution {}".format(res_type))

            if res_type == 'orig':
                filename = "{0}_{1}_mjysr.fits".format(gal.lower(), name)
            else:
                filename = "{0}_{1}_{2}_mjysr.fits".format(gal.lower(), name, res_type)

            if not os.path.exists(osjoin(data_path, gal, filename)):
                print("Could not find {}. Skipping".format(filename))
                continue

            hdu = fits.open(osjoin(data_path, gal, filename))
            proj = Projection.from_hdu(fits.PrimaryHDU(hdu[0].data.squeeze(),
                                                       hdu[0].header))
            # Attach equiv Gaussian beam
            if res_type == 'orig':
                proj = proj.with_beam(fitinfo_dict[gal][name]['beam'])
            # The convolved images should have all have a beam saved

            # Take minimal shape. Remove empty space.
            # Erode edges to avoid noisier region/uneven scans
            mask = np.isfinite(proj)
            mask = nd.binary_erosion(mask, np.ones((3, 3)), iterations=8)

            # Add radial box cut
            # radius = gal_cls.radius(header=proj.header).to(u.kpc)
            # rad_mask = radius < cut
            # mask = np.logical_and(mask, rad_mask)

            # Pick out region determined from the column density map extents
            # PLus some padding at the edge
            spat_mask = spat_mask_maker(*proj.spatial_coordinate_map)

            proj = proj[nd.find_objects(mask & spat_mask)[0]]

            # Save the cut-out, if it doesn't already exist
            out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))

            if not os.path.exists(osjoin(data_path, gal, out_filename)):
                proj.write(osjoin(data_path, gal, out_filename))

            # look at each image.
            if img_view:
                proj.quicklook()
                plt.draw()
                input("{}".format(filename))
                plt.close()

            if res_type == 'orig':
                save_name = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)
            else:
                save_name = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(), name, res_type)

            # For now skip already saved power-spectra
            if os.path.exists(osjoin(data_path, gal, save_name)) and skip_check:
                print("Already saved pspec for {}. Skipping".format(filename))
                continue
            else:
                os.system("rm -f {}".format(osjoin(data_path, gal, save_name)))

            pspec = PowerSpectrum(proj, distance=dist)
            pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                      high_cut=0.1 / u.pix,
                      use_pyfftw=True, threads=ncores,
                      apod_kern=fitinfo_dict[gal][name]['apod_kern'])

            pspec.save_results(osjoin(data_path, gal, save_name),
                               keep_data=False)

            del pspec, proj, hdu
