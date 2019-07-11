
'''
Make the power-spectrum fit table for the col dens maps.
'''

import os
import pandas as pd
from astropy import units as u


osjoin = os.path.join
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

df = pd.read_csv(osjoin(data_path, "pspec_fit_results.csv"),
                 index_col=0)

params = ['logA', 'ind', 'logB']

ress = ['orig', 'mod']

distances = {'lmc': 50.1 * u.kpc,
             'smc': 62.1 * u.kpc,
             'm33': 840 * u.kpc,
             'm31': 744 * u.kpc}


fitinfo_dict = dict()

fitinfo_dict["lmc"] = \
    {'mips24': {'orig_beam': 6.5, 'conv_beam': 11.0,
                'apod_kern': False,
                'use_beam': True},
     'mips70': {'orig_beam': 18.7, 'conv_beam': 30.0,
                'apod_kern': False,
                'use_beam': True},
     'pacs100': {'orig_beam': 7.1, 'conv_beam': 9.0,
                 'apod_kern': False,
                 'use_beam': True},
     'mips160': {'orig_beam': 38.8, 'conv_beam': 64.0,
                 'apod_kern': False,
                 'use_beam': True},
     'pacs160': {'orig_beam': 11.2, 'conv_beam': 14.0,
                 'apod_kern': False,
                 'use_beam': True},
     'spire250': {'orig_beam': 18.2, 'conv_beam': 21.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire350': {'orig_beam': 25, 'conv_beam': 28.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire500': {'orig_beam': 36.4, 'conv_beam': 41.0,
                  'apod_kern': False,
                  'use_beam': True}}

fitinfo_dict["smc"] = \
    {'mips24': {'orig_beam': 6.5, 'conv_beam': 11.0,
                'apod_kern': False,
                'use_beam': True},
     'mips70': {'orig_beam': 18.7, 'conv_beam': 30.0,
                'apod_kern': False,
                'use_beam': False},
     'pacs100': {'orig_beam': 7.1, 'conv_beam': 9.0,
                 'apod_kern': False,
                 'use_beam': True},
     'mips160': {'orig_beam': 38.8, 'conv_beam': 64.0,
                 'apod_kern': True,
                 'use_beam': True},
     'pacs160': {'orig_beam': 11.2, 'conv_beam': 14.0,
                 'apod_kern': True,
                 'use_beam': False},
     'spire250': {'orig_beam': 18.2, 'conv_beam': 21.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire350': {'orig_beam': 25, 'conv_beam': 28.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire500': {'orig_beam': 36.4, 'conv_beam': 41.0,
                  'apod_kern': False,
                  'use_beam': True}}

fitinfo_dict["m33"] = \
    {'mips24': {'orig_beam': 6.5, 'conv_beam': 11.0,
                'apod_kern': False,
                'use_beam': True},
     'mips70': {'orig_beam': 18.7, 'conv_beam': 30.0,
                'apod_kern': False,
                'use_beam': True},
     'pacs100': {'orig_beam': 7.1, 'conv_beam': 9.0,
                 'apod_kern': False,
                 'use_beam': True},
     'mips160': {'orig_beam': 38.8, 'conv_beam': 64.0,
                 'apod_kern': False,
                 'use_beam': True},
     'pacs160': {'orig_beam': 11.2, 'conv_beam': 14.0,
                 'apod_kern': False,
                 'use_beam': True},
     'spire250': {'orig_beam': 18.2, 'conv_beam': 21.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire350': {'orig_beam': 25, 'conv_beam': 28.0,
                  'apod_kern': False,
                  'use_beam': True},
     'spire500': {'orig_beam': 36.4, 'conv_beam': 41.0,
                  'apod_kern': False,
                  'use_beam': True}}

fitinfo_dict["m31"] = \
    {'mips24': {'orig_beam': 6.5, 'conv_beam': 11.0,
                'apod_kern': False,
                'use_beam': True},
     'mips70': {'orig_beam': 18.7, 'conv_beam': 30.0,
                'apod_kern': False,
                'use_beam': True},
     'pacs100': {'orig_beam': 7.1, 'conv_beam': 9.0,
                 'apod_kern': False,
                 'use_beam': False},
     'mips160': {'orig_beam': 38.8, 'conv_beam': 64.0,
                 'apod_kern': True,
                 'use_beam': True},
     'pacs160': {'orig_beam': 11.2, 'conv_beam': 14.0,
                 'apod_kern': True,
                 'use_beam': True},
     'spire250': {'orig_beam': 18.2, 'conv_beam': 21.0,
                  'apod_kern': True,
                  'use_beam': True},
     'spire350': {'orig_beam': 25, 'conv_beam': 28.0,
                  'apod_kern': True,
                  'use_beam': True},
     'spire500': {'orig_beam': 36.4, 'conv_beam': 41.0,
                  'apod_kern': True,
                  'use_beam': True}}

nice_bands = {'mips24': "MIPS 24",
              'mips70': "MIPS 70",
              'pacs100': "PACS 100",
              'mips160': "MIPS 160",
              'pacs160': "PACS 160",
              'spire250': "SPIRE 250",
              'spire350': "SPIRE 350",
              'spire500': "SPIRE 500",
              }

# Open a new tex file
out_name = "table_pspec_fits.tex"

out_string = r"\\begin{tabular}{ccccccc} \n"

# out_string += r" \\diagbox{Original}{Convolved} & Band & Resolution ($\arcsec$) & Phys. Resolution (pc) & log$_{10}$ $A$ & $\beta$ & log$_{10}$ B \\\  \hline \n")
out_string += r" Galaxy & Band & Resolution ($\arcsec$) & Phys. Resolution (pc) & log$_{10}$ $A$ & $\beta$ & log$_{10}$ B \\\  \hline \n"


# \usepackage{ amssymb }
special_symbs = [r'\\bigstar', r'\\blacklozenge']

# Galaxy
for gal in ['lmc', 'smc', 'm31', 'm33']:

    # Band
    for i, band in enumerate(fitinfo_dict[gal]):

        data_orig = df.loc[f"{gal}_{band}_orig"]
        data_conv = df.loc[f"{gal}_{band}_mod"]

        if i == 0:
            gal_string = r" {} & ".format(gal.upper())
        else:
            gal_string = r"    & "

        gal_string_conv = r"     & "

        # Attach PSF use and apodization to the band here
        band_str = nice_bands[band]

        use_psf = fitinfo_dict[gal][band]['use_beam']
        use_apod = fitinfo_dict[gal][band]['apod_kern']
        if not use_psf and use_apod:
            band_str += r"$^{\\bigstar,\\blacklozenge}$"
        elif not use_psf:
            band_str += r"$^{\\bigstar}$"
        elif use_apod:
            band_str += r"$^{\\blacklozenge}$"

        gal_string += band_str + " & "
        gal_string_conv += "     & "

        res = fitinfo_dict[gal][band]['orig_beam']

        phys_res = (res * u.arcsec).to(u.rad).value * distances[gal].to(u.pc)
        phys_res = round(phys_res.value)

        res_conv = fitinfo_dict[gal][band]['conv_beam']

        phys_res_conv = \
            (res_conv * u.arcsec).to(u.rad).value * distances[gal].to(u.pc)
        phys_res_conv = round(phys_res_conv.value)

        # Data resolution
        # gal_string += f"\\diagbox{{{res}}}{{{res_conv}}} & "
        # gal_string += f"\\diagbox{{{phys_res}}}{{{phys_res_conv}}} & "

        gal_string += f" {res} & "
        gal_string += f" {phys_res} & "
        gal_string_conv += f" {res_conv} & "
        gal_string_conv += f" {phys_res_conv} & "


        for par in params:
            val_orig = data_orig[par]
            err_orig = data_orig[par + "_std"]

            val_conv = data_conv[par]
            err_conv = data_conv[par + "_std"]

            # par_str = f"\\diagbox{{${val_orig:.2f}\pm{err_orig:.2f}$}}{{${val_conv:.2f}\pm{err_conv:.2f}$}} & "
            par_str = f"${val_orig:.2f}\pm{err_orig:.2f}$ & "
            par_str_conv = f"${val_conv:.2f}\pm{err_conv:.2f}$ & "

            gal_string += par_str
            gal_string_conv += par_str_conv

        gal_string = gal_string[:-3] + " \\\ "
        gal_string_conv = gal_string_conv[:-3] + " \\\ "

        out_string += gal_string + " \n"
        out_string += gal_string_conv + " \n"

out_string += r"\\end{tabular}"

with open(osjoin(data_path, out_name), 'w') as tfile:

    print(out_string, file=tfile)

# Copy to the paper folder
paper_folder = os.path.expanduser("~/ownCloud/My_Papers/In\ Prep/LG_dust_powerspectra/")

os.system("cp {0} {1}".format(osjoin(data_path, out_name),
                              paper_folder))
