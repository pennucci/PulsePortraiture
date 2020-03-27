#! /usr/bin/env python

import os

telescope_code_dict = dict()

if 'TEMPO2' in os.environ:
    path_to_obs = os.path.join(os.environ['TEMPO2'], 'observatory', \
                               'observatories.dat')
    if os.path.isfile(path_to_obs):
        obs_dat = open(path_to_obs, 'r').readlines()
        for line in obs_dat:
            if line.startswith('#') or line.startswith('\n'):
                pass
            else:
                line = line.split()
                telescope = line[-2].upper()
                short_code = line[-1]
                telescope_code_dict[telescope] = [short_code]
    path_to_aliases = os.path.join(os.environ['TEMPO2'], 'observatory', \
                                   'aliases')
    if os.path.isfile(path_to_aliases):
        aliases = open(path_to_aliases, 'r').readlines()
        for line in aliases:
            if line.startswith('#') or line.startswith('\n'):
                pass
            else:
                line = line.split()
                for telescope, short_code in list(telescope_code_dict.items()):
                    if line[0] == short_code[0]:
                        for alias in line[1:]:
                            telescope_code_dict[telescope].append(alias)

else:
    telescope_code_dict = {'ARECIBO': ['ao', '3', 'arecebo', 'arecibo'],
                           'AXIS': ['axi'],
                           'CAMBRIDGE': ['cam'],
                           'COE': ['coe'],
                           'DARNHALL': ['l'],
                           'DE601': ['EFlfr'],
                           'DE601HBA': ['EFlfrhba'],
                           'DE601LBA': ['EFlfrlba'],
                           'DE601LBH': ['EFlfrlbh'],
                           'DE602': ['UWlfr'],
                           'DE602HBA': ['UWlfrhba'],
                           'DE602LBA': ['UWlfrlba'],
                           'DE602LBH': ['UWlfrlbh'],
                           'DE603': ['TBlfr'],
                           'DE603HBA': ['TBlfrhba'],
                           'DE603LBA': ['TBlfrlba'],
                           'DE603LBH': ['TBlfrlbh'],
                           'DE604': ['POlfr'],
                           'DE604HBA': ['POlfrhba'],
                           'DE604LBA': ['POlfrlba'],
                           'DE604LBH': ['POlfrlbh'],
                           'DE605': ['JUlfr'],
                           'DE605HBA': ['JUlfrhba'],
                           'DE605LBA': ['JUlfrlba'],
                           'DE605LBH': ['JUlfrlbh'],
                           'DE609': ['NDlfr'],
                           'DE609HBA': ['NDlfrhba'],
                           'DE609LBA': ['NDlfrlba'],
                           'DE609LBH': ['NDlfrlbh'],
                           'DEFFORD': ['n'],
                           'DSS_43': ['tid43', '6'],
                           'EFFELSBERG': ['eff', 'g'],
                           'EFFELSBERG_ASTERIX': ['effix'],
                           'FAST': ['fast'],
                           'FI609': ['Filfr'],
                           'FI609HBA': ['Filfrhba'],
                           'FI609LBA': ['Filfrlba'],
                           'FI609LBH': ['Filfrlbh'],
                           'FR606': ['FRlfr'],
                           'FR606HBA': ['FRlfrhba'],
                           'FR606LBA': ['FRlfrlba'],
                           'FR606LBH': ['FRlfrlbh'],
                           'GB140': ['gb140'],
                           'GB300': ['gb300'],
                           'GB853': ['gb853'],
                           'GBT': ['gbt', '1', 'gb'],
                           'GEO600': ['geo600'],
                           'GMRT': ['gmrt'],
                           'GOLDSTONE': ['gs'],
                           'GRAO': ['grao'],
                           'HAMBURG': ['hamburg'],
                           'HANFORD': ['lho'],
                           'HARTEBEESTHOEK': ['hart'],
                           'HOBART': ['hob'],
                           'JBOAFB': ['jbafb'],
                           'JBODFB': ['jbdfb', 'q'],
                           'JBOROACH': ['jbroach'],
                           'JB_42FT': ['jb42'],
                           'JB_MKII': ['jbmk2', 'h'],
                           'JB_MKII_DFB': ['jbmk2dfb'],
                           'JB_MKII_RCH': ['jbmk2roach'],
                           'JODRELL': ['jb', '8', 'y', 'z'],
                           'JODRELL2': ['q'],
                           'JODRELLM4': ['jbm4'],
                           'KAGRA': ['kagra'],
                           'KAT-7': ['k7'],
                           'KNOCKIN': ['m'],
                           'LA_PALMA': ['p'],
                           'LIVINGSTON': ['llo'],
                           'LOFAR': ['lofar', 't'],
                           'LWA1': ['lwa1', 'x'],
                           'MEERKAT': ['meerkat', 'm'],
                           'MKIII': ['jbmk3', 'j'],
                           'MOST': ['mo'],
                           'MWA': ['mwa'],
                           'NANCAY': ['ncy', 'f'],
                           'NANSHAN': ['NS'],
                           'NARRABRI': ['atca', '2'],
                           'NUPPI': ['ncyobs', 'w'],
                           'OP': ['obspm'],
                           'PARKES': ['pks', '7'],
                           'PRINCETON': ['princeton'],
                           'SE607': ['ONlfr'],
                           'SE607HBA': ['ONlfrhba'],
                           'SE607LBA': ['ONlfrlba'],
                           'SE607LBH': ['ONlfrlbh'],
                           'SRT': ['srt', 'z'],
                           'STL_BAT': ['STL_BAT'],
                           'TABLEY': ['k'],
                           'UAO': ['NS'],
                           'UK608': ['UKlfr'],
                           'UK608HBA': ['UKlfrhba'],
                           'UK608LBA': ['UKlfrlba'],
                           'UK608LBH': ['UKlfrlbh'],
                           'UTR-2': ['UTR2'],
                           'VIRGO': ['virgo'],
                           'VLA': ['vla', 'c'],
                           'WARKWORTH_12M': ['wark12m'],
                           'WARKWORTH_30M': ['wark30m'],
                           'WSRT': ['wsrt', 'i']}

# if 'TEMPO' in os.environ:
#    path_to_obsys = os.path.join(os.environ['TEMPO'],'obsys.dat')
#    if os.path.isfile(path_to_obsys):
#        obsys_dat = open(path_to_obsys,'r').readlines()
