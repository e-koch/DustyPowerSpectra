
'''
Define one dictionary to set the fit restrictions used in fit_pspec.py
'''

from radio_beam import Beam
from astropy import units as u

fitinfo_dict = dict()

fitinfo_dict["LMC"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec),
                'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'mips70': {'beam': Beam(18.7 * u.arcsec),
                'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs100': {'beam': Beam(7.1 * u.arcsec),
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True},
     'mips160': {'beam': Beam(38.8 * u.arcsec),
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True},
     'pacs160': {'beam': Beam(11.2 * u.arcsec),
                 'low_cut': None, 'high_cut': None,
                 'use_beam': True},
     'spire250': {'beam': Beam(18.2 * u.arcsec),
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True},
     'spire350': {'beam': Beam(25 * u.arcsec),
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True},
     'spire500': {'beam': Beam(36.4 * u.arcsec),
                  'low_cut': None, 'high_cut': None,
                  'use_beam': True}}

fitinfo_dict["SMC"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'low_cut': None, 'high_cut': 0.06,
                'use_beam': False},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'low_cut': None, 'high_cut': None,
                 'use_beam': True},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'low_cut': None, 'high_cut': 0.1,
                 'use_beam': False},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'low_cut': None, 'high_cut': None,
                  'use_beam': True},
     'spire350': {'beam': Beam(25 * u.arcsec), 'low_cut': None, 'high_cut': None,
                  'use_beam': True},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'low_cut': None, 'high_cut': None,
                  'use_beam': True}}

fitinfo_dict["M33"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire350': {'beam': Beam(25 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True}}

fitinfo_dict["M31"] = \
    {'mips24': {'beam': Beam(6.5 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'mips70': {'beam': Beam(18.7 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs100': {'beam': Beam(7.1 * u.arcsec), 'low_cut': None, 'high_cut': 0.03,
                'use_beam': False},
     'mips160': {'beam': Beam(38.8 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'pacs160': {'beam': Beam(11.2 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire250': {'beam': Beam(18.2 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire350': {'beam': Beam(25 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True},
     'spire500': {'beam': Beam(36.4 * u.arcsec), 'low_cut': None, 'high_cut': None,
                'use_beam': True}}