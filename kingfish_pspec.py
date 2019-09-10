
'''
Run through the Herschel maps of the KINGFISH sample

Just running PACS for now.

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
from radio_beam import Beam
import skimage.morphology as mo

from turbustat.statistics import PowerSpectrum
from turbustat.statistics.psds import make_radial_freq_arrays


osjoin = os.path.join


data_path = os.path.expanduser("~/bigdata/erosolow/surveys/kingfish/kingfish_pacs_scanam_v17/")
plot_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/kingfish_plots")

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# Grab all names out of the sample

fits_files = glob(osjoin(data_path, "*.fits"))

gal_names = np.unique([os.path.split(fname)[-1].split("_")[0] for fname in fits_files])

# Loop through. Common structure for all bands

fitinfo_dict = \
    {'pacs100': {'beam': Beam(7.1 * u.arcsec), 'apod_kern': None,
                 'filename_suffix': "scanamorphos_v16.9_pacs100_0.fits",
                 'low_int_cut': None, 'high_int_cut': None},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'apod_kern': None,
                 'filename_suffix': "scanamorphos_v16.9_pacs160_0.fits",
                 'low_int_cut': None, 'high_int_cut': None},}
     # 'spire250': {'beam': Beam(18.2 * u.arcsec), 'apod_kern': None,
     #              'filename_suffix': "_kingfish_spire250_v3-0_scan.fits",
     #              'low_int_cut': None, 'high_int_cut': None},
     # 'spire350': {'beam': Beam(25 * u.arcsec), 'apod_kern': None,
     #              'filename_suffix': "_kingfish_spire350_v3-0_scan.fits",
     #              'low_int_cut': None, 'high_int_cut': None},
     # 'spire500': {'beam': Beam(36.4 * u.arcsec), 'apod_kern': None,
     #              'filename_suffix': "_kingfish_spire500_v3-0_scan.fits",
     #              'low_int_cut': None, 'high_int_cut': None}}


for gal in gal_names:

    print(f"On galaxy {gal}")

    fig, axs = plt.subplots(2, 3)

    fig2 = plt.figure()
    one_ax = fig2.add_subplot(111)

    for name, ax in zip(fitinfo_dict, axs):

        filename = f"{gal}_{fitinfo_dict[name]['filename_suffix']}"
        hdu = fits.open(osjoin(data_path, filename))

        # Multiple images in the PACS imgs
        if 'pacs' in name:
            data = hdu[0].data[0]
        else:
            data = hdu[0].data

        proj = Projection.from_hdu(fits.PrimaryHDU(data,
                                                   hdu[0].header))
        # Attach equiv Gaussian beam
        proj = proj.with_beam(fitinfo_dict[name]['beam'])
        # The convolved images should have all have a beam saved

        # Take minimal shape. Remove empty space.
        # Erode edges to avoid noisier region/uneven scans
        mask = np.isfinite(proj)
        mask = nd.binary_erosion(mask, np.ones((3, 3)), iterations=45)

        proj = proj[nd.find_objects(mask)[0]]

        # Save the cut-out, if it doesn't already exist
        # out_filename = "{}_cutout.fits".format(filename.rstrip(".fits"))

        # if not os.path.exists(osjoin(data_path, 'raw', out_filename)):
        #     proj.write(osjoin(data_path, 'raw', out_filename))

        # look at each image.
        # if img_view:
        #     proj.quicklook()
        #     plt.draw()
        #     input("{}".format(filename))
        #     plt.close()

        # if res_type == 'orig':
        #     save_name = "{0}_{1}_mjysr.pspec.pkl".format(gal.lower(), name)
        # else:
        #     save_name = "{0}_{1}_{2}_mjysr.pspec.pkl".format(gal.lower(), name, res_type)

        # For now skip already saved power-spectra
        # if os.path.exists(osjoin(data_path, 'raw', save_name)) and skip_check:
        #     print("Already saved pspec for {}. Skipping".format(filename))
        #     continue
        # else:
        #     os.system("rm -f {}".format(osjoin(data_path, 'raw', save_name)))

        pspec = PowerSpectrum(proj)  # , distance=dist)
        pspec.run(verbose=False, beam_correct=False, fit_2D=False,
                  high_cut=0.1 / u.pix,
                  use_pyfftw=False, threads=1,)
                  # apodize_kernel=fitinfo_dict[name]['apod_kern'])

        # Plot 2D spec, 1D pspec, img

        im0 = ax[0].imshow(np.log10(pspec.ps2D), origin='lower', cmap='viridis')
        fig.colorbar(im0, ax=ax[0])

        # Convert to angular units
        xunit = u.arcsec**-1
        ang_freqs = pspec._spatial_freq_unit_conversion(pspec.freqs, xunit).value

        ax[1].loglog(ang_freqs, pspec.ps1D)
        ax[1].set_xlabel("Spat freq (1/arcsec)")
        ax[1].set_title(name)

        major = fitinfo_dict[name]['beam'].major.to(u.arcsec)
        ax[1].axvline(1 / major.value, linestyle='--', color='k')

        im2 = ax[2].imshow(proj.value, origin='lower', cmap='viridis',
                           vmax=np.nanpercentile(proj.value, 99),
                           vmin=np.nanpercentile(proj.value, 10))
        fig.colorbar(im2, ax=ax[2])

        one_ax.loglog(ang_freqs, pspec.ps1D, label=name)
        one_ax.set_xlabel("Spat freq (1/arcsec)")
        one_ax.axvline(1 / major.value, linestyle='--', color='k')

    # Show the 1D pspec on a common scale
    axs[0][1].get_shared_x_axes().join(axs[0][1], axs[1][1])

    one_ax.legend(frameon=True)

    fig.savefig(osjoin(plot_path, f"{gal}_pacs100_160_pspec.png"))
    fig.savefig(osjoin(plot_path, f"{gal}_pacs100_160_pspec.pdf"))
    plt.close()

    fig2.savefig(osjoin(plot_path, f"{gal}_pacs100_160_1Dpspec.png"))
    fig2.savefig(osjoin(plot_path, f"{gal}_pacs100_160_1Dpspec.pdf"))
    plt.close()
