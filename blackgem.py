# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:59:29 2024

@author: xGeeRe
"""

import os
from astropy.table import Table
from astropy.time import Time
# from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
# from astropy.io import fits
# import astropy.units as u
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.transforms as transforms
# import seaborn as sns
# from astropy.timeseries import LombScargle
import coord_utils # my own
# from astroplan import EclipsingSystem
# from astroplan import FixedTarget, Observer, is_observable, is_event_observable, PeriodicEvent
# from astroplan import PhaseConstraint, AtNightConstraint, AltitudeConstraint, LocalTimeConstraint
# import datetime as dt
# import tglc
# from tglc.quick_lc import tglc_lc
# import pickle
# import lightkurve as lk
from lightcurve_utils import plot_lightcurves

def separate_bg_lc():
    table_coord = Table.read('data/70_targets.csv', format='ascii.csv')
    # Create directory where to store the output files
    output_path = './BG_lightcurves_std'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # Create directory where to store the separated NEOWISE light curves
    pathos = './BG_lightcurves'
    if not os.path.isdir(pathos):
        os.makedirs(pathos)
    
    # Bulk downloading ZTF light curves from IRSA generates a single table with all the sources
    # So we separate the sources in independent tables/files
    # MeerLICHT_bulk_file = pd.read_csv('data/bg_out.csv') # Read the file with all the sources' light curves
    MeerLICHT_bulk_file =Table.read('data/bg_out.csv', format='ascii.csv')
    
    # Clean the data removing points with high error on the magnitude or the flux, negative flux, 
    # with high chi/N values or with background magnitude lower than the observation
    MeerLICHT_bulk_file = MeerLICHT_bulk_file[(MeerLICHT_bulk_file['SNR_ZOGY']>5) & (MeerLICHT_bulk_file['MAGERR_ZOGY']<1) & 
                            (MeerLICHT_bulk_file['LIMMAG_ZOGY']>MeerLICHT_bulk_file['MAG_ZOGY'])]
    
    for name in list(set(MeerLICHT_bulk_file['NUMBER_IN'])):
        separated_lc = MeerLICHT_bulk_file[MeerLICHT_bulk_file['NUMBER_IN']==name]
        # Xmatch with the asciicoords file
        column_names = ['RA', 'DEC', 'ra', 'dec', 'DR3_source_id']
        xmatch_table = coord_utils.sky_xmatch(separated_lc, table_coord, 1, column_names)
        file_name = xmatch_table['Name'][0]
        separated_lc = separated_lc.to_pandas()
        separated_lc.to_csv(f'{pathos}/{file_name}.csv')

    mydtype=[("name", "|S25"), ("inst","|S15"), ("filter","|S25"), ("mjd","<f8"),("bjd","<f8"), 
             ("mag","<f4"), ("magerr","<f4"), ("flux","<f4"), ("fluxerr","<f4")]
    ins = 'BG'
    
    # Loop for every light curve file in the lc directory
    lc = os.listdir(pathos)
    for file in lc:
        # Read light curve file
        table_clean = Table.read(f'{pathos}/{file}', format='ascii.csv')
                    
        # Create and fill output table
        final_table = np.zeros(len(table_clean), dtype=mydtype)
        
        file_name = file.split('.')[0]
        coords = SkyCoord(table_coord['ra'][table_coord['DR3_source_id']==int(file_name)],
                          table_coord['dec'][table_coord['DR3_source_id']==int(file_name)],
                          unit='deg', frame='icrs')
        location = EarthLocation.of_site('lasilla')
        mjd_time = Time(table_clean['MJD-OBS'], format='mjd', 
                        scale='utc', location=location)
        ltt_bary = mjd_time.light_travel_time(coords)
        time_barycentre = mjd_time.tdb + ltt_bary        
        
        final_table['inst'] = ins
        final_table['filter'] = table_clean['FILTER']
        final_table['mjd'] = table_clean['MJD-OBS']
        final_table['bjd'] = time_barycentre.value
        final_table['mag'] = table_clean['MAG_ZOGY']
        final_table['magerr'] = table_clean['MAGERR_ZOGY']
        final_table['name'] = file_name
        
        final_table = Table(final_table)
        
        # Write the output table in the desired directory
        name = final_table['name'][0]
        final_table.write(f'{output_path}/{name}.csv', format='ascii.csv', overwrite=True)

#######
# plot_lightcurves('BG_lightcurves_std', savefig=False, plot=True, rang='single')

#######
def ztf_variability():
    table_coord = Table.read('data/70_targets.csv', format='ascii.csv')
    ztf_fields = os.listdir('data/01_00_prediction_xgb_dnn_fields')
    
    for file in ztf_fields:
        ztf_xmatch = Table.read(f'data/01_00_prediction_xgb_dnn_fields/{file}', format='ascii.csv')
        column_names = ['ra', 'dec', 'ra', 'dec', 'DR3_source_id']
        xmatch_table = coord_utils.sky_xmatch(ztf_xmatch, table_coord, 3, column_names)
        matches = xmatch_table[xmatch_table['Name']!='']
        num_matches = len(matches)
        matches.write(f'data/ztf_var_matches/xmatch_{num_matches}_{file}', format='ascii.csv', overwrite=True)
        
        
        
        