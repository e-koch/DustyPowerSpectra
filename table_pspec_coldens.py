
'''
Make the power-spectrum fit table for the col dens maps.
'''

import os
import pandas as pd


osjoin = os.path.join
data_path = os.path.expanduser("~/bigdata/ekoch/Utomo19_LGdust/")

df = pd.read_csv(osjoin(data_path, "pspec_coldens_fit_results.csv"),
                 index_col=0)

params = ['logA', 'ind', 'logB', 'logC']

# Open a new tex file
out_name = "table_pspec_coldens.tex"

out_string = r"\begin{tabular}{ccccccc} \n"

out_string += r" Galaxy & Resolution ($\arcsec$) & Phys. Resolution (pc) & log$_{10}$ $A$ & $\beta$ & log$_{10}$ B & log$_{10}$ C \\  \hline \n"

for gal, res, phys_res in zip(['lmc', 'smc', 'm31', 'm33'],
                              [53.4, 43.2, 46.3, 41.0],
                              [13, 13, 167, 167]):

    data = df.loc[gal]

    gal_string = r" {} & ".format(gal.upper())

    # Data resolution
    gal_string += r"{} & ".format(res)
    gal_string += r"{} & ".format(phys_res)

    for par in params:
        gal_string += r"${0:.2f}\pm{1:.2f}$ & ".format(data[par],
                                                    data["{}_std".format(par)])

    gal_string = gal_string[:-3] + " \\"

    out_string += gal_string + " \n"

out_string += r" \end{tabular}"

with open(osjoin(data_path, out_name), 'w') as tfile:

    print(out_string, file=tfile)

# Copy to the paper folder
paper_folder = os.path.expanduser("~/ownCloud/My_Papers/In\ Prep/LG_dust_powerspectra/")

os.system("cp {0} {1}".format(osjoin(data_path, out_name),
                              paper_folder))
