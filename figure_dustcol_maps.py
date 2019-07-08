
'''
Make two column + two row figure showing the regions used for all
power-spectra.
'''

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from radio_beam import Beam
import seaborn as sb
import astropy.units as u
import os
from os.path import join as osjoin
import scipy.ndimage as nd

from plotting_styles import twocolumn_figure


twocolumn_figure()

gals = {'LMC': 50.1 * u.kpc, 'SMC': 62.1 * u.kpc,
        'M31': 744 * u.kpc, 'M33': 840 * u.kpc}

# Running on SegFault w/ data on bigdata
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

# Make a plot output folder
plot_folder = osjoin(data_path, "summary_plots")
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

fitinfo_dict = dict()

fitinfo_dict["LMC"] = {'filename': r"lmc_dust.surface.density_FB.beta=1.8_gauss53.4_regrid_mwsub.fits",
                       'beam': Beam(53.4 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 50.1 * u.kpc}
fitinfo_dict["SMC"] = {'filename': r"smc_dust.surface.density_FB.beta=1.8_gauss43.2_regrid_mwsub.fits",
                       'beam': Beam(43.2 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 62.1 * u.kpc}
fitinfo_dict["M31"] = {'filename': r"m31_dust.surface.density_FB.beta=1.8_gauss46.3_regrid_bksub.fits",
                       'beam': Beam(46.3 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 744 * u.kpc}
fitinfo_dict["M33"] = {'filename': r"m33_dust.surface.density_FB.beta=1.8_gauss41.0_regrid_bksub.fits",
                       'beam': Beam(41.0 * u.arcsec), 'apod_kern': None,
                       'low_int_cut': None, 'high_int_cut': None,
                       'low_cut': None, 'high_cut': None,
                       'distance': 840 * u.kpc}

fig, axes = plt.subplots(2, 2)

vmax_scale = 0.4

cmap = plt.cm.binary
cmap.set_bad(color='white')

for gal, ax in zip(fitinfo_dict, axes.ravel()):


    filename = osjoin(data_path, gal, fitinfo_dict[gal]['filename'])

    hdu_coldens = fits.open(filename)

    # The edges of the maps have high uncertainty. For M31, this may be
    # altering the shape of the power-spectrum. Try removing these edges:
    coldens_mask = np.isfinite(hdu_coldens[0].data[0].squeeze())

    coldens_mask = nd.binary_erosion(coldens_mask,
                                     structure=np.ones((3, 3)),
                                     iterations=8)

    # Get minimal size
    masked_data = hdu_coldens[0].data[0].squeeze()
    masked_data[~coldens_mask] = np.NaN

    proj_coldens = hdu_coldens[0].data[0][nd.find_objects(coldens_mask)[0]]

    # proj_coldens = proj_coldens.with_beam(fitinfo_dict[gal]['beam'])

    # Pad to have equal shape for the plot
    bigger_ax = np.argmax(proj_coldens.shape)
    smaller_ax = np.argmin(proj_coldens.shape)

    diff_size = proj_coldens.shape[bigger_ax] - proj_coldens.shape[smaller_ax]

    if smaller_ax == 0:
        padder = [(diff_size // 2, diff_size // 2), (0, 0)]
    else:
        padder = [(0, 0), (diff_size // 2, diff_size // 2)]

    proj_coldens_padded = np.pad(proj_coldens, padder, mode='constant',
                                 constant_values=np.NaN)

    proj_coldens_padded[np.isnan(proj_coldens_padded)] = 0.

    im = ax.imshow(proj_coldens_padded,
                   cmap=cmap,
                   origin='lower',
                   interpolation='nearest',
                   norm=ImageNormalize(vmin=0.0,
                                       vmax=vmax_scale,
                                       stretch=AsinhStretch()),)
    ax.set_xticks([])
    ax.set_yticks([])

    length = (1000. * u.pc / fitinfo_dict[gal]['distance']).to(u.deg, u.dimensionless_angles())
    length_pix = length.value / np.abs(hdu_coldens[0].header['CDELT2'])

    x_vals = [0.1 * proj_coldens_padded.shape[0]] * 2
    y_val1 = 0.1 * proj_coldens_padded.shape[0]
    y_val2 = 0.1 * proj_coldens_padded.shape[0] + length_pix

    ax.plot(x_vals, [y_val1, y_val2], 'k',
            linewidth=2)
    ax.text(x_vals[0], 0.07 * proj_coldens_padded.shape[0],
            "1 kpc", color='k', va='top', ha='center')

    # Galaxy name
    if gal == 'M33' or gal == 'M31':
        y_val_gal = 0.9 * proj_coldens_padded.shape[0]
    else:
        y_val_gal = y_val1

    ax.text(0.86 * proj_coldens_padded.shape[0], y_val_gal, gal,
            color='k', va='center', ha='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
# Insert a colourbar on the left

plt.tight_layout()

cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.set_label(r"$\Sigma_{\rm dust}$ (M$_{\odot}$ pc$^{-2}$)")

fig.savefig(osjoin(plot_folder, "Sigma_dust_maps.png"))
fig.savefig(osjoin(plot_folder, "Sigma_dust_maps.pdf"))
plt.close()
