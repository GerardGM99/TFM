# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:13:20 2024

@author: Gerard Garcia
"""

import os
from astropy.table import Table
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.io import fits
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import seaborn as sns
from astropy.timeseries import LombScargle
import coord_utils # my own
from astroplan import EclipsingSystem
from astroplan import FixedTarget, Observer, is_observable, is_event_observable, PeriodicEvent
from astroplan import PhaseConstraint, AtNightConstraint, AltitudeConstraint, LocalTimeConstraint
import datetime as dt
import tglc
from tglc.quick_lc import tglc_lc
import pickle
import lightkurve as lk

def standard_table(ins, lc_directory, asciicords, output_format='csv', suffix='csv'):
    '''
    Given data from an specified catalog/instrument, creates a table with 
    relevant data for plotting light curves, with columns:
        - name: name of the object (e.g. the Gaia DR3 ID)
        - inst: the catalog/instrument from which the data is taken
        - filter: the band used by the instrument
        - mjd: the Modified Julian Date of the observation
        - bjd: the Barycentric Dynamical Time of the observation
        - mag: the magnitude
        - magerr: the error on the mag
        - flux
        - fluxerr: the error on the flux
    IMPORTANT: OUTLIERS ARE NOT REMOVED!
    
    Parameters
    ----------
    ins: string
        Catalog/instrument from which the data is taken.
    lc_directory: string
        Directory with the data files.
    asciicords: string
        File with the coordinates and the id of all the sources.
    output_format: string, optional
        The format of the output tables (csv, latex...).
    suffix: string, optional
        Suffix of the output files (csv, tex...).

    Returns
    -------
    None

    '''
    
    # Format for the output table
    mydtype=[("name", "|S25"), ("inst","|S15"), ("filter","|S25"), ("mjd","<f8"),("bjd","<f8"), 
             ("mag","<f4"), ("magerr","<f4"), ("flux","<f4"), ("fluxerr","<f4")]
    
    # Read directory with light curve files
    if (ins != 'NEOWISE') and (ins != 'IRSA_ZTF') and (ins != 'MeerLICHT'):
        lc = os.listdir(lc_directory)
    
    # Read the file with the ids and coordinates of the sources
    #table_coord = pd.read_csv(asciicords)
    table_coord = Table.read(asciicords, format='ascii.csv')
        
    # Light curves from TESS
    if ins == 'TESS':
        # Create directory where to store the output files
        output_path = './TESS_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Loop for every light curve file in the lc directory    
        for file in lc:
            # Read light curve file
            table = Table.read(f'{lc_directory}/{file}', format='ascii.csv')
            
            auth = file.split('_')[2].split('.')[0]
            name = file.split('_')[0]
            
            if auth=='QLP': # https://archive.stsci.edu/hlsp/qlp
                table_clean = table[(table['quality']==0)]
                time = Time(table_clean['time'] + 2457000., format='jd', scale='tdb')
                try:
                    flux = table_clean['kspsap_flux']
                    fluxerr = table_clean['kspsap_flux_err']
                except:
                    flux = table_clean['det_flux']
                    fluxerr = table_clean['det_flux_err']
            elif auth=='SPOC': # https://archive.stsci.edu/hlsp/tess-spoc
                table_clean = table[(table['quality']==0)]
                time = Time(table_clean['time'] + 2457000., format='jd', scale='tdb')
                time = Time(time, format='jd', scale='tcb')
                flux = table_clean['pdcsap_flux']
                fluxerr = table_clean['pdcsap_flux_err']
            elif auth=='TASOC': # https://archive.stsci.edu/hlsp/tasoc
                table_clean = table[(table['quality']==0)]
                time = Time(table_clean['time'] + 2457000., format='jd', scale='tdb')
                flux = table_clean['flux_corr']
                fluxerr = table_clean['flux_corr_err']
            elif auth=='CDIPS': # https://archive.stsci.edu/hlsp/cdips
                table_clean = table[(table['irq2']=='G')]
                time = Time(table_clean['time'], format='jd', scale='tdb')
                flux = table_clean['ifl2']
                fluxerr = table_clean['ife2']
                mag = table_clean['irm2']
                magerr = table_clean['ire2']
            elif auth=='TGLC': # https://archive.stsci.edu/hlsp/tglc
                table_clean = table[(table['quality']==0)]
                time = Time(table_clean['time'] + 2457000., format='jd', scale='tdb')
                flux = table_clean['cal_psf_flux']
                fluxerr = np.zeros(len(table_clean))
            elif auth=='GSFC-ELEANOR-LITE': # https://archive.stsci.edu/hlsp/gsfc-eleanor-lite
                table_clean = table[(table['quality']==0)]
                time = Time(table_clean['time'] + 2457000., format='jd', scale='tdb')
                flux = table_clean['corr_flux']
                fluxerr = np.zeros(len(table_clean))
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = auth
            final_table['bjd'] = time.mjd
            final_table['flux'] = flux
            final_table['fluxerr'] = fluxerr
            if auth=='CDIPS':
                final_table['mag'] = mag
                final_table['magerr'] = magerr
            final_table['name'] = name
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            file_name = file.split('.')[0]
            final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
    
    # Light curves from Zwicky Transient Facility [ZTF] (https://www.ztf.caltech.edu)
    if ins == 'ZTF':
        # Create directory where to store the output files
        output_path = './ZTF_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Loop for every light curve file in the lc directory    
        for file in lc:
            # Read light curve file
            table = Table.read(f'{lc_directory}/{file}', format='ascii.csv')
            
            # Clean the data (https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_extended_cautionary_notes.pdf)
            table_clean = table[(table['catflags']==0) & (table['chi']<4)]
            
            # Xmatch with the asciicoords file
            column_names = ['ra', 'dec', 'ra', 'dec', 'DR3_source_id']
            xmatch_table = coord_utils.sky_xmatch(table_clean, table_coord, 1, column_names)
            file_name = xmatch_table['Name'][0]
            
            coords = SkyCoord(table_coord['ra'][table_coord['DR3_source_id']==int(file_name)],
                              table_coord['dec'][table_coord['DR3_source_id']==int(file_name)],
                              unit='deg', frame='icrs')
            location = EarthLocation.of_site('Palomar')
            mjd_time = Time(table_clean['mjd'], format='mjd', 
                            scale='utc', location=location)
            ltt_bary = mjd_time.light_travel_time(coords)
            time_barycentre = mjd_time.tdb + ltt_bary
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['filtercode']
            final_table['mjd'] = mjd_time.value
            final_table['bjd'] = time_barycentre.value
            final_table['mag'] = table_clean['mag']
            final_table['magerr'] = table_clean['magerr']
            final_table['name'] = file_name
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
            
    
    if ins == 'IRSA_ZTF':
        # Create directory where to store the output files
        output_path = './IRSA_ZTF_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Create directory where to store the separated light curves
        pathos = './IRSA_ZTF_lightcurves'
        if not os.path.isdir(pathos):
            os.makedirs(pathos)
        
        # Bulk downloading ZTF light curves from IRSA generates a single table with all the sources
        # So we separate the sources in independent tables/files
        irsa_ztf_bulk_file = pd.read_csv(lc_directory) # Read the file with all the sources' light curves
        
        # Clean the data (https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_extended_cautionary_notes.pdf)
        irsa_ztf_bulk_file = irsa_ztf_bulk_file[(irsa_ztf_bulk_file['catflags']==0) & (irsa_ztf_bulk_file['chi']<4)]
        irsa_ztf_bulk_file = Table.from_pandas(irsa_ztf_bulk_file)
        
        # Xmatch with the asciicoords file
        column_names = ['ra', 'dec', 'ra', 'dec', 'DR3_source_id']
        xmatch_table = coord_utils.sky_xmatch(irsa_ztf_bulk_file, table_coord, 1, column_names)
        if len(set(xmatch_table['Name']))>1: #Not needed if there's only one source
            for name in set(xmatch_table['Name']):
                if name != '':
                    separated_lc = xmatch_table[xmatch_table['Name']==name]
                    separated_lc.write(f'{pathos}/{name}.csv', overwrite=True)
        
        # Loop for every light curve file in the lc directory
        lc = os.listdir(pathos)
        for file in lc:
            # Read light curve file
            table_clean = Table.read(f'{pathos}/{file}', format='ascii.csv')
            name = table_clean['Name'][0]
            
            coords = SkyCoord(table_coord['ra'][table_coord['DR3_source_id']==name],
                              table_coord['dec'][table_coord['DR3_source_id']==name],
                              unit='deg', frame='icrs')
            location = EarthLocation.of_site('Palomar')
            mjd_time = Time(table_clean['mjd'], format='mjd', 
                            scale='utc', location=location)
            ltt_bary = mjd_time.light_travel_time(coords)
            time_barycentre = mjd_time.tdb + ltt_bary
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['filtercode']
            final_table['mjd'] = mjd_time.value
            final_table['bjd'] = time_barycentre.value
            final_table['mag'] = table_clean['mag']
            final_table['magerr'] = table_clean['magerr']
            final_table['name'] = table_clean['Name']
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            final_table.write(f'{output_path}/{name}.{suffix}', format=output_format, overwrite=True)
            
    
    # Light curves from All-Sky Automated Surveey for Supernovae [ASAS-SN] (https://asas-sn.osu.edu)   
    if ins == 'ASAS-SN':
          # Create directory where to store the output files
          output_path = './ASAS-SN_lightcurves_std'
          if not os.path.isdir(output_path):
              os.makedirs(output_path)
          
          # Loop for every light curve file in the lc directory
          for file in lc:
              # Read light curve file
              table = Table.read(f'{lc_directory}/{file}', format='ascii.csv')
              
              # Clean the data (bad observations have mag_err = 99.99; quality flag can be G (good) and B (bad))
              table_clean = table[(table['mag_err']<99) & (table['quality']=='G')]
              
              file_name_only = file.split('.csv')[0]
              coords = SkyCoord(table_coord['ra'][table_coord['DR3_source_id']==int(file_name_only)],
                                table_coord['dec'][table_coord['DR3_source_id']==int(file_name_only)],
                                unit='deg', frame='icrs')
              location = EarthLocation.of_site('ctio')
              jd_time = Time(table_clean['jd'], format='jd', 
                              scale='utc', location=location)
              jd_time.format = 'mjd'
              ltt_bary = jd_time.light_travel_time(coords)
              time_barycentre = jd_time.tdb + ltt_bary
              
              # Create and fill output table
              final_table = np.zeros(len(table_clean), dtype=mydtype)
              
              final_table['inst'] = ins
              final_table['filter'] = table_clean['phot_filter']
              final_table['mjd'] = jd_time.value
              final_table['bjd'] = time_barycentre.value
              final_table['mag'] = table_clean['mag']
              final_table['magerr'] = table_clean['mag_err']
              final_table['flux'] = table_clean['flux']
              final_table['fluxerr'] = table_clean['flux_err']
              
              # Name of the file, the same as the imput lc file because the downloaded tables from asas-sn don't have coordinates
              final_table['name'] = np.repeat(file_name_only, len(final_table))
              
              final_table = Table(final_table)
              
              # Write the output table in the desired directory
              final_table.write(f'{output_path}/{file_name_only}.{suffix}', format=output_format, overwrite=True)
              
    
    # Light curves from Asteroid Terrestrial-impact Last Alert System [ATLAS] (https://fallingstar-data.com/forcedphot/)
    if ins == 'ATLAS':
        # Create directory where to store the output files
        output_path = './ATLAS_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Loop for every light curve file in the lc directory    
        for file in lc:
            # Read light curve file
            table = Table.read(f'{lc_directory}/{file}', format='ascii')
            
            # Clean the data removing points with high error on the magnitude or the flux, negative flux, 
            # with high chi/N values or with background magnitude lower than the observation
            table_clean = table[(table['dm']<=0.05) & (table['duJy']<3000) & (table['uJy']>0) & (table['chi/N']<100) & (table['mag5sig']>table['m'])]
            table_clean = table_clean[(table_clean['F']=='o')|(table_clean['F']=='c')]
            
            # Xmatch with the asciicoords file
            column_names = ['RA', 'Dec', 'ra', 'dec', 'DR3_source_id']
            xmatch_table = coord_utils.sky_xmatch(table_clean, table_coord, 1, column_names)
            file_name = xmatch_table['Name'][0]
            
            coords = SkyCoord(table_coord['ra'][table_coord['DR3_source_id']==int(file_name)],
                              table_coord['dec'][table_coord['DR3_source_id']==int(file_name)],
                              unit='deg', frame='icrs')
            location = EarthLocation.of_site('haleakala')
            mjd_time = Time(table_clean['##MJD'], format='mjd', 
                            scale='utc', location=location)
            ltt_bary = mjd_time.light_travel_time(coords)
            time_barycentre = mjd_time.tdb + ltt_bary
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['F']
            final_table['mjd'] = mjd_time.value
            final_table['bjd'] = time_barycentre.value
            final_table['mag'] = table_clean['m']
            final_table['magerr'] = table_clean['dm']
            final_table['flux'] = table_clean['uJy']
            final_table['fluxerr'] = table_clean['duJy']
            final_table['name'] = file_name
                                   
            # Give a name to each file acording to the asciicords file which has the id and the coordinates of the sources
            # ra = round(table_clean['RA'][0], 2)
            # dec = round(table_clean['Dec'][0], 2)
            # file_name = table_coord[source_name][(round(table_coord['ra'],2)==ra) & (round(table_coord['dec'],2)==dec)].iloc[0]
            # final_table['name'] = np.repeat(file_name, len(final_table))
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
    
    
    if ins == 'NEOWISE':
        # Create directory where to store the output files
        output_path = './NEOWISE_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Create directory where to store the separated NEOWISE light curves
        pathos = './NEOWISE_lightcurves'
        if not os.path.isdir(pathos):
            os.makedirs(pathos)
                        
        # Bulk downloading NEOWISE light curves from IRSA generates a single table with all the sources
        # So we separate the sources in independent tables/files
        neowise_bulk_file = pd.read_csv(lc_directory)
        
        # Clean the data using the signal to noise ratio and chi^2
        # (https://wise2.ipac.caltech.edu/docs/release/neowise/expsup/sec2_1a.html)        
        neowise_bulk_file = neowise_bulk_file[(neowise_bulk_file['w1snr']>2) & (neowise_bulk_file['w2snr']>2) &
                                              (neowise_bulk_file['w1rchi2']<150) & (neowise_bulk_file['w2rchi2']<150)]
        neowise_bulk_file = Table.from_pandas(neowise_bulk_file)
        
        # Xmatch with the asciicoords file
        column_names = ['ra_01', 'dec_01', 'ra', 'dec', 'DR3_source_id']
        xmatch_table = coord_utils.sky_xmatch(neowise_bulk_file, table_coord, 1, column_names)
        if len(set(xmatch_table['Name']))>1: #Not needed if there's only one source
            for name in set(xmatch_table['Name']):
                if name != '':
                    separated_lc = xmatch_table[xmatch_table['Name']==name]
                    separated_lc.write(f'{pathos}/{name}.csv', overwrite=True)
        
        # Loop for every light curve file in the lc directory
        lc = os.listdir(pathos)
        for file in lc:
            # Read light curve file
            table_clean = Table.read(f'{pathos}/{file}', format='ascii.csv')
            
            # Clean the data using cc_flags flag (Contamination and confusion flags) and ph_qual flag (Photometric quality flag)
            # (https://wise2.ipac.caltech.edu/docs/release/neowise/expsup/sec2_1a.html)
            # cc_flags = 0000; ph_qual = A, B or C; [qual_frame > 0 could be used as well]
            # ph_qual_list = ['AA','AB','AC','BA','BB','BC','CA','CB','CC']
            # table_clean = table[(table['cc_flags']=='0000')]
            # ph_qual_mask = np.array([(qual in ph_qual_list) for qual in table_clean['ph_qual']])
            # table_clean = table_clean[ph_qual_mask]
            
            # Separate the two filters, W1 and W2           
            time_obs = table_clean['mjd']
            mag_obs_W1 = table_clean['w1mpro']
            mag_obs_err_W1 = table_clean['w1sigmpro']
            mag_obs_W2 = table_clean['w2mpro']
            mag_obs_err_W2 = table_clean['w2sigmpro']
            
            
            # Create and fill output table
            final_table = np.zeros(len(mag_obs_W1)+len(mag_obs_W2), dtype=mydtype)
            
            final_table['inst'] = ins

            final_table['filter'][0:len(mag_obs_W1)] = 'W1' 
            final_table['mjd'][0:len(mag_obs_W1)] = time_obs
            final_table['mag'][0:len(mag_obs_W1)] =  mag_obs_W1
            final_table['magerr'][0:len(mag_obs_W1)] = mag_obs_err_W1
            
            final_table['filter'][len(mag_obs_W1):] = 'W2' 
            final_table['mjd'][len(mag_obs_W1):] = time_obs
            final_table['mag'][len(mag_obs_W1):] =  mag_obs_W2
            final_table['magerr'][len(mag_obs_W1):] = mag_obs_err_W2
            
            file_name = table_clean['Name'][0]
            final_table['name'] = file_name
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True) 
            
    # Light curves from BlackGEM (https://www.eso.org/public/teles-instr/lasilla/blackgem/)
    if ins == 'BLACKGEM':
        # Create directory where to store the output files
        output_path = './BLACKGEM_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            
        # Loop for every light curve file in the lc directory    
        for file in lc:
            # Read light curve file
            table = Table.read(f'{lc_directory}/{file}', format='ascii.csv')
                
            if len(table)>0:
                # Clean the data removing points with high error on the magnitude or the flux, negative flux, 
                # with high chi/N values or with background magnitude lower than the observation
                table_clean = table[(table['SNR_OPT']>5) & (table['MAGERR_OPT']<1) & 
                                        (table['LIMMAG_OPT']>table['MAG_OPT'])]
                
                if len(table_clean)>0:
                    # Create and fill output table
                    final_table = np.zeros(len(table_clean), dtype=mydtype)
                    
                    final_table['name'] = table_clean['SOURCE_ID']
                    final_table['inst'] = ins
                    final_table['filter'] = table_clean['filter']
                    final_table['mjd'] = table_clean['mjd']
                    final_table['bjd'] = Time(final_table['mjd'], format='mjd').tdb.value
                    final_table['mag'] = table_clean['MAG_OPT']
                    final_table['magerr'] = table_clean['MAGERR_OPT']
                        
                    final_table = Table(final_table)
                    file_name = table_clean['SOURCE_ID'][0]
                        
                    # Write the output table in the desired directory
                    final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
                    
    if ins == 'MeerLICHT':
        # Create directory where to store the output files
        output_path = './MeerLICHT_lightcurves_std'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Create directory where to store the separated NEOWISE light curves
        pathos = './MeerLICHT_lightcurves'
        if not os.path.isdir(pathos):
            os.makedirs(pathos)
        
        # Bulk downloading ZTF light curves from IRSA generates a single table with all the sources
        # So we separate the sources in independent tables/files
        MeerLICHT_bulk_file = pd.read_csv(lc_directory) # Read the file with all the sources' light curves
        
        # Clean the data removing points with high error on the magnitude or the flux, negative flux, 
        # with high chi/N values or with background magnitude lower than the observation
        MeerLICHT_bulk_file = MeerLICHT_bulk_file[(MeerLICHT_bulk_file['SNR_OPT']>5) & (MeerLICHT_bulk_file['MAGERR_OPT']<1) & 
                                (MeerLICHT_bulk_file['LIMMAG_OPT']>MeerLICHT_bulk_file['MAG_OPT'])]
        
        for name in list(set(MeerLICHT_bulk_file['DR3_source_id'])):
            separated_lc = MeerLICHT_bulk_file[MeerLICHT_bulk_file['DR3_source_id']==name]
            separated_lc.to_csv(f'{pathos}/{name}.csv')
            
        # Loop for every light curve file in the lc directory
        lc = os.listdir(pathos)
        for file in lc:
            # Read light curve file
            table_clean = Table.read(f'{pathos}/{file}', format='ascii.csv')
                        
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['FILTER']
            final_table['mjd'] = table_clean['MJD-OBS']
            final_table['bjd'] = Time(final_table['mjd'], format='mjd').tdb.value
            final_table['mag'] = table_clean['MAG_OPT']
            final_table['magerr'] = table_clean['MAGERR_OPT']
            final_table['name'] = table_clean['DR3_source_id']
            
            final_table = Table(final_table)
            
            # Write the output table in the desired directory
            name = final_table['name'][0]
            final_table.write(f'{output_path}/{name}.{suffix}', format=output_format, overwrite=True)
            
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def tess_lglc(ide, target, save_dir):
    '''
    ABANDONDED
    '''
    # target: TIC ID (preferred, 'TIC 12345678'), Target ID ('TOI 519') or coordinates ('ra dec')
    local_directory = f'{save_dir}/{ide}/'    # directory to save all files
    os.makedirs(local_directory, exist_ok=True)
    
#   tglc_lc(target=target, 
#         local_directory=local_directory, 
#         size=50, # FFI cutsize. Recommand at least 50 or larger for better performance. Cannot exceed 99. 
#                  # Downloading FFI might take longer (or even cause timeouterror) for larger sizes. 
#         save_aper=True, # whether to save 5*5 pixels timeseries of the decontaminated images in fits file primary HDU
#         limit_mag=13, # the TESS magnitude lower limit of stars to output
#         get_all_lc=False, # whether to return all lcs in the region. If False, return the nearest star to the target coordinate
#         first_sector_only=True, # whether to return only lcs from the sector this target was first observed. 
#                                 # If False, return all sectors of the target, but too many sectors could be slow to download.
#         last_sector_only=False, # whether to return only lcs from the sector this target was last observed. 
#         sector=None, # If first_sector_only = True or last_sector_only = True and type(sector) != int, return first or last sector.
#                      # If first(last)_sector_only=False and sector = None, return all observed sectors
#                      # If first(last)_sector_only=False and type(sector) == int, return only selected sector. 
#                      # (Make sure only put observed sectors. All available sectors are printed in the sector table.)
#         prior=None)  # If None, does not allow all field stars to float. SUGGESTED for first use. 
#                      # If float (usually <1), allow field stars to float with a Gaussian prior with the mean 
#                      # at the Gaia predicted value the width of the prior value multiplied on the Gaia predicted value.
    
    tglc_lc(target=target, local_directory=local_directory, size=40, save_aper=True, limit_mag=13, 
            get_all_lc=False, first_sector_only=False, sector=None, prior=None)
    

def TESSlk_get_lcs(table):
    '''
    Extracts the processed TESS light curves using lightkurve, for the objects
    in the given table. Saves the light curve as a csv named 
    {Gaia DR3 ID}_S{TESS sector}_{processing pipeline}.csv
    
    Parameters
    ----------
    table: pandas.DataFrame or astropy.Table
        Table with the Gaia DR3 ID (column must be named 'DR3_source_id'), the 
        TIC (column must be named 'TIC').
    '''
    
    outdir = './TESS_lightcurves'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for name, tic in zip(table['DR3_source_id'], table['TIC']):
        search_result = lk.search_lightcurve('TIC %s' %tic)
        lc = search_result.download_all()
        if lc is not None:
            for i in range(len(lc)):
                try:
                    lc_clean = lc[i].remove_nans().remove_outliers()
                except:
                    lc_clean = lc[i]
                author = lc_clean.author
                sector = search_result[i].mission[0].split(' ')[2]
                lc_clean.to_csv(f'{outdir}/{name}_S{sector}_{author}.csv', overwrite=True)
    
    
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def remove_outliers(y, method='median', n_std=3):
    '''
    Removes outliers from magnitude data.
    
    Parameters
    ----------
    y: array of floats
        Magnitude values from which outliers are removed
    method: string, optional
        Method used to remove outliers. Either 'median' (removes points a 
        number of standard deviations away from the median), 'iqr' 
        (InterQuartile Range) or 'eb' (removes points that exceed 6 times the 
        IQR at the faint side, and 2 times at the bright side), or None to not 
        remove any outliers. The default is 'median'.
    n_std: float, optional
        Number of standard deviations or IQRs away from the median to reject the
        outliers.

    Returns
    -------
    Sigma: array of booleans
        Mask to remove outliers from a table.

    '''
    # Remove outliers...
    # ...with median and standard deviation
    if method=='median':
        median = np.median(y) 
        std = np.std(y)
        sigma = np.abs(y- median) < n_std*std
                        
    # ...with InterQuartile Range
    if method=='iqr':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        median = np.median(y)
        iqr = q3 - q1
        sigma = np.abs(y- median) < n_std*iqr
    
    # 10 times on the faint side and 2 times on the bright side, to account for
    # eclipses
    if method=='eb':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        median = np.median(y)
        iqr = q3 - q1
        sigma = ((y-median > 0) & (y-median < 10*iqr)) | ((y-median < 0) & (median-y < 2*iqr))
        
    if method==None:
        sigma = np.repeat(True, len(y))
    
    return sigma


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def bin_lightcurve(x, y, yerr=None, mode="average", binsize=3):
    '''
    Bins the lightcurve in bins of binsize.
    '''

    if binsize < 0:
        return x, y, yerr
    
    if yerr is None:
        yerr = np.zeros_like(y)
    
    if np.ndim(x) == 0:
        x = np.array([x])
        y = np.array([y])
        yerr = np.array([yerr])
    elif len(x) == 0:
        return np.array([]), np.array([]), np.array([])
        
        
    x_i = list(set(np.round(x/binsize) * binsize))
    x_i.sort()
    x_i = np.array(x_i)
    y_o = []
    yerr_o = []
    x_o = []
    xerr_o  = []
    

    for xi in x_i:
        mask = np.abs(x-xi) < binsize/2.
        mask = mask * ~(np.isnan(yerr))
        #If we only have one register, no binning needed
        if len(x[mask])==1:
            x_o.append(x[mask][0])
            y_o.append(y[mask][0])
            xerr_o.append(0)

            if not yerr is None:
                yerr_o.append(yerr[mask][0])
            else:
                yerr_o.append([0])
        elif np.any(mask):
            #print ("OBJECTS PER EPOCH", len(y[mask]))
           
            x_o.append(np.average(x[mask]))
            xerr_o.append(np.std(x[mask]))
            #print(np.std(x[mask]))
            if mode == "median":
                y_o.append(np.median(y[mask]))
                if not yerr is None:
                    #yerr_o.append(np.sqrt(np.sum((yerr[mask])**2 ))/len(yerr[mask]))
                    yerr_o.append(np.sqrt(np.sum((y[mask]-np.median(y[mask]))**2 ))/len(y[mask]))
            else:
                fluxes = 10**(-0.4*y[mask])
                fluxerr = fluxes - 10**(-0.4*(y[mask] + yerr[mask]))
                #fluxarr = np.array([ufloat(z1,z2) for z1,z2 in zip(fluxes, fluxerr)])
                               
                #Weighted mean:
                #https://ned.ipac.caltech.edu/level5/Leo/Stats4_5.html
                avgflux = np.nansum(fluxes/fluxerr**2) / np.nansum(1./fluxerr**2)
                
                #avgmag = -2.5 * np.log10(fluxarr.mean().n)
                avgmag = -2.5 * np.log10(avgflux)
                stdflux = np.sqrt(1./ np.nansum(1./fluxerr**2))
                stdev = 2.5*stdflux/(avgflux*np.log(10))               
                #stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + stdflux))
                
                # stdev_fl = np.std(fluxes[~np.isnan(fluxes)])
                # stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + stdev_fl))
                
                #print (yerr[mask], len(fluxerr), stdev)
                
                #stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + fluxarr.mean().s))
                
                y_o.append(avgmag)
                yerr_o.append(stdev)

    x_o = np.array(x_o)
    xerr_o = np.array(xerr_o)

    y_o = np.array(y_o)
    yerr_o = np.array(yerr_o)
    
    return x_o, y_o, xerr_o, yerr_o


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def plot_lightcurves(lc_dir, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, 
                     rang=None, plot=False, savefig=True):
    '''
    Plots the light curves in the given directory. The light curve files MUST
    be in the format given by the function lightcurve_utils.standard_table. 
    Iterates the 'lightcurve_utils.plot_ind_lightcurves' for all the files in 
    the given directory.
    
    Parameters
    ----------
    lc_dir: string
        Directory where the light curve files are stored
    ref_mjd: float, optional
        Reference time given that the imput times are in MJD. The 58519.45
        that is setup coincides with the start of ZTF observations.
    y_ax: string, optional
        If the y axis is in magnitude or flux. Eihter 'mag' or 'flux'.
    outliers: string, optional
        The manner in which the outliers in magnitude/flux are removed. Either
        'iqr' (InterQuartile Range) or 'median' (None to not remove outliers).
    binsize: int, optional
        If greater than 0, bins the ligthcurve in bins with size binsize.
    rang: tuple of floats, otional
        The min and max values for the time range to plot. If None, all the 
        data points are ploted. If 'single' the plots are done around each
        night with data points.
    plot: boolean, optional
        If True, shows the plots
    safefig: boolean, optional
        If True, safe the figure in the directory '{ins}_plots' (e.g. plots for
        ZTF light curves would be stored in 'ZTF_plots')
    
    Returns
    -------
    None

    '''
    
    lc_file = os.listdir(lc_dir)

    if (rang is None) or (type(rang)==tuple):
        for ide in lc_file:
            plot_ind_lightcurves(f'{lc_dir}/{ide}', ref_mjd=ref_mjd, y_ax=y_ax, outliers=outliers, n_std=n_std, 
                                 binsize=binsize, rang=rang, plot=plot, savefig=savefig)
            
    if rang=='single':
        for ide in lc_file:
            file = Table.read(f'{lc_dir}/{ide}', format='ascii.csv')
            nights = list(set(file['mjd'].astype(int)))
            nights=sorted(nights)
            ranges=[(x,x+1) for x in nights]
            for t_min, t_max in ranges:
                plot_ind_lightcurves(f'{lc_dir}/{ide}', ref_mjd=ref_mjd, y_ax=y_ax, outliers=outliers, n_std=n_std, 
                                     binsize=binsize, rang=(t_min,t_max), plot=True, savefig=False)
        
        
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


# def plot_ind_lightcurves(file_name, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, rang=None, plot=False, savefig=True):
#     '''
#     Plots the light curve of the given file. The light curve files MUST
#     be in the format given by the function lightcurve_utils.standard_table.
#     If the data files has less than 5 points nothing is drawn and a message os
#     returned.
    
#     Parameters
#     ----------
#     file: string
#         File with the light curve data
#     ref_mjd: float, optional
#         Reference time given that the imput times are in MJD. The 58519.45
#         that is setup coincides with the start of ZTF observations.
#     y_ax: string, optional
#         If the y axis is in magnitude or flux. Eihter 'mag' or 'flux'.
#     outliers: string, optional
#         The manner in which the outliers in magnitude/flux are removed. Either
#         'iqr' (InterQuartile Range) or 'median' (None to not remove outliers).
#     binsize: int, optional
#         If greater than 0, bins the ligthcurve in bins with size binsize.
#     rang: tuple of floats, otional
#         The min and max values for the time range to plot. If None, all the 
#         data points are ploted
#     plot: boolean, optional
#         If True, shows the plot
#     safefig: boolean, optional
#         If True, safe the figure in the directory '{ins}_plots' (e.g. plots for
#         ZTF light curves would be stored in 'ZTF_plots')
    
#     Returns
#     -------
#     None

#     '''
    
#     # Dictionary of colors for each instrument/filter combination
#     color_dict = {
#         ('ZTF', 'zg'):'g',
#         ('ZTF', 'zr'):'r',
#         ('ZTF', 'zi'):'goldenrod',
#         ('IRSA_ZTF', 'zg'):'g',
#         ('IRSA_ZTF', 'zr'):'r',
#         ('IRSA_ZTF', 'zi'):'gold',
#         ('ASAS-SN', 'V'):'darkcyan',
#         ('ASAS-SN', 'g'):'blue',
#         ('ATLAS', 'o'):'orange',
#         ('ATLAS', 'c'):'cyan',
#         ('NEOWISE', 'W1'):'darkred',
#         ('NEOWISE', 'W2'):'slategray',
#         ('BLACKGEM', 'u'):'purple',
#         ('BLACKGEM', 'g'):'skyblue',
#         ('BLACKGEM', 'r'):'orange',
#         ('BLACKGEM', 'i'):'firebrick',
#         ('BLACKGEM', 'z'):'sienna',
#         ('BLACKGEM', 'q'):'black',
#         ('MeerLICHT', 'u'):'purple',
#         ('MeerLICHT', 'g'):'skyblue',
#         ('MeerLICHT', 'r'):'orange',
#         ('MeerLICHT', 'i'):'firebrick',
#         ('MeerLICHT', 'z'):'sienna',
#         ('MeerLICHT', 'q'):'black',
#         ('TESS', 'QLP'):'magenta',
#         ('TESS', 'SPOC'):'magenta',
#         ('TESS', 'TASOC'):'magenta',
#         ('TESS', 'CDIPS'):'magenta',
#         ('TESS', 'TGLC'):'magenta',
#         ('TESS', 'GSFC-ELEANOR-LITE'):'magenta'}
    
#     plt.ioff()
#     #data_files_names_only = file_name.split(".csv")[0]
#     # Read file
#     file = Table.read(file_name, format='ascii.csv')
#     if len(file)>4:
#         ins = file['inst'][0]
#         source_name = file['name'][0]
#         out_path = f'{ins}_plots'
#         if not os.path.isdir(out_path):
#             os.makedirs(out_path)
            
#         if ins=='NEOWISE':
#             ms=6
#             cs=4
#             ew=4
#         else:
#             ms=3
#             cs=2
#             ew=2
            
#         # Create figure and axes
#         fig, ax = plt.subplots(constrained_layout=True)
#         fig.set_size_inches(9,5.5)
#         ax.grid(alpha=0.5)
            
#         # Plot the light curve for each filter
#         available_filters=list(set(file['filter']))
#         available_filters=sorted(available_filters)
#         for band in available_filters:
#             if y_ax=='mag':
#                 # Remove outliers
#                 #sigma = remove_outliers(file['mag'][file['filter']==band], method=outliers, n_std=n_std)
                                    
#                 # Select time, magnitudes and magnitude errors
#                 t_observed = np.array(file["bjd"][file['filter']==band])#[sigma]
#                 y_observed = np.array(file['mag'][file['filter']==band])#[sigma]
#                 uncert = np.array(file["magerr"][file['filter']==band])#[sigma]
                    
#                 # Bin the data
#                 if binsize > 0:
#                     t_observed, y_observed, t_err, uncert =  bin_lightcurve(t_observed, y_observed, yerr=uncert, 
#                                                                             binsize=binsize, mode="average")
#                 else:
#                     t_err=None
                        
#                 # Select only a set limited range of points to plot
#                 if rang is not None:
#                     mask = (t_observed > rang[0]) & (t_observed < rang[1])
#                     t_observed = t_observed[mask]
#                     y_observed = y_observed[mask]
#                     uncert = uncert[mask]
#                     if binsize > 0:
#                         t_err = t_err[mask]
                    
#                 # Plot light curve
#                 plot_color = color_dict.get((ins,band), 'black')
#                 ax.errorbar(t_observed - ref_mjd, y_observed, xerr=t_err, yerr=uncert, color=plot_color, label=band, 
#                             fmt = "o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
#                 ax.set_ylabel("Magnitude", family = "serif", fontsize = 16)
                    
#             if y_ax=='flux':
#                 # Remove outliers
#                 sigma = remove_outliers(file['flux'][file['filter']==band], method=outliers)
                    
#                 # Select time, magnitudes and magnitude errors
#                 t_observed = np.array(file["bjd"][file['filter']==band])[sigma]
#                 y_observed = np.array(file['flux'][file['filter']==band])[sigma]
#                 uncert = np.array(file["fluxerr"][file['filter']==band])[sigma]
                
#                 # Select only a set limited range of points to plot
#                 if rang is not None:
#                     mask = (t_observed > rang[0]) & (t_observed < rang[1])
#                     t_observed = t_observed[mask]
#                     y_observed = y_observed[mask]
#                     uncert = uncert[mask]
                    
#                 # Plot light curve
#                 plot_color = color_dict.get((ins,band), 'black')
#                 ax.errorbar(t_observed - ref_mjd, y_observed, yerr=uncert, color=plot_color, label=band, 
#                             fmt = "o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
#                 ax.set_ylabel("Flux", family = "serif", fontsize = 16)
                    
                    
#         ax.set_title(f"{source_name}", weight = "bold") # Gaia DR3 
#         ax.set_xlabel(f"MJD - {ref_mjd} [days]", family = "serif", fontsize = 16)
#         ax.tick_params(which='major', width=2, direction='out')
#         ax.tick_params(which='major', length=7)
#         ax.tick_params(which='minor', length=4, direction='out')
#         ax.tick_params(labelsize = 16)
#         ax.minorticks_on()
#         ax.invert_yaxis()
#         xmin, xmax = ax.get_xlim()    
        
#         ref_mjd = 58190.45
#         def xconv(x):
#             tyear = Time(x+ref_mjd, format="mjd")
#             return tyear.jyear # days to years
#         xmin2 = xconv(xmin)
#         xmax2 = xconv(xmax)
        
#         ay2 = plt.twiny()
#         ay2.minorticks_on()
#         ay2.tick_params(which='major', width=2, direction='out')
#         ay2.tick_params(which='major', length=7)
#         ay2.tick_params(which='minor', length=4, direction='out')
#         ay2.tick_params(labelsize = 16)
#         ay2.set_xlim([xmin2, xmax2])
#         ay2.set_xlabel('Year', fontsize=16)
#         ay2.ticklabel_format(useOffset=False)
            
                
#         colors_for_filters = [color_dict.get((ins, filt), None) for filt in available_filters]
                
#         ax.legend(loc = "upper right", ncols = len(available_filters), labelcolor = colors_for_filters, 
#                   shadow = True, columnspacing = 0.8, handletextpad = 0.5, handlelength = 1, markerscale = 1.5, 
#                   prop = {"weight" : "bold"})
        
#         if savefig == True:
#             plt.savefig(f'{out_path}/{source_name}.png', bbox_inches = "tight", format = "png")
        
#         if plot == True:
#             plt.show()
        
#         plt.close()
        
#     else:
#         print('Less than 5 points in the time series for source '+str(file['name'][0]))


def plot_ind_lightcurves(file_name, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, rang=None, plot=False, savefig=True, ax=None):
    '''
    Plots the light curve of the given file. The light curve files MUST
    be in the format given by the function lightcurve_utils.standard_table.
    If the data files has less than 5 points nothing is drawn and a message is
    returned.
    
    Parameters
    ----------
    file_name: string
        File with the light curve data
    ref_mjd: float, optional
        Reference time given that the input times are in MJD. The 58519.45
        that is setup coincides with the start of ZTF observations.
    y_ax: string, optional
        If the y axis is in magnitude or flux. Either 'mag' or 'flux'.
    outliers: string, optional
        The manner in which the outliers in magnitude/flux are removed. Either
        'iqr' (InterQuartile Range) or 'median' (None to not remove outliers).
    binsize: int, optional
        If greater than 0, bins the lightcurve in bins with size binsize.
    rang: tuple of floats, optional
        The min and max values for the time range to plot. If None, all the 
        data points are plotted
    plot: boolean, optional
        If True, shows the plot
    savefig: boolean, optional
        If True, saves the figure in the directory '{ins}_plots' (e.g. plots for
        ZTF light curves would be stored in 'ZTF_plots')
    ax: matplotlib.axes._subplots.AxesSubplot, optional
        The Axes object to plot on. If None, a new figure is created. The default is None.
    
    Returns
    -------
    None
    '''
    
    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF', 'zg'):'g',
        ('ZTF', 'zr'):'r',
        ('ZTF', 'zi'):'goldenrod',
        ('IRSA_ZTF', 'zg'):'g',
        ('IRSA_ZTF', 'zr'):'r',
        ('IRSA_ZTF', 'zi'):'gold',
        ('ASAS-SN', 'V'):'darkcyan',
        ('ASAS-SN', 'g'):'blue',
        ('ATLAS', 'o'):'orange',
        ('ATLAS', 'c'):'cyan',
        ('NEOWISE', 'W1'):'darkred',
        ('NEOWISE', 'W2'):'slategray',
        ('BLACKGEM', 'u'):'purple',
        ('BLACKGEM', 'g'):'skyblue',
        ('BLACKGEM', 'r'):'orange',
        ('BLACKGEM', 'i'):'firebrick',
        ('BLACKGEM', 'z'):'sienna',
        ('BLACKGEM', 'q'):'black',
        ('MeerLICHT', 'u'):'purple',
        ('MeerLICHT', 'g'):'skyblue',
        ('MeerLICHT', 'r'):'orange',
        ('MeerLICHT', 'i'):'firebrick',
        ('MeerLICHT', 'z'):'sienna',
        ('MeerLICHT', 'q'):'black',
        ('TESS', 'QLP'):'magenta',
        ('TESS', 'SPOC'):'magenta',
        ('TESS', 'TASOC'):'magenta',
        ('TESS', 'CDIPS'):'magenta',
        ('TESS', 'TGLC'):'magenta',
        ('TESS', 'GSFC-ELEANOR-LITE'):'magenta'}
    
    plt.ioff()
    file = Table.read(file_name, format='ascii.csv')
    if len(file) > 4:
        ins = file['inst'][0]
        source_name = file['name'][0]
        out_path = f'{ins}_plots'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            
        if ins == 'NEOWISE':
            ms = 6
            cs = 4
            ew = 4
        else:
            ms = 3
            cs = 2
            ew = 2
            
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
            fig.set_size_inches(9, 5.5)
            ax.grid(alpha=0.5)
            
        available_filters = sorted(set(file['filter']))
        for band in available_filters:
            if y_ax == 'mag':
                if ins == 'NEOWISE':
                    t_observed = np.array(file["mjd"][file['filter'] == band])
                else:
                    t_observed = np.array(file["bjd"][file['filter'] == band])
                y_observed = np.array(file['mag'][file['filter'] == band])
                uncert = np.array(file["magerr"][file['filter'] == band])
                    
                if binsize > 0:
                    t_observed, y_observed, t_err, uncert = bin_lightcurve(t_observed, y_observed, yerr=uncert, 
                                                                            binsize=binsize, mode="average")
                else:
                    t_err = None
                        
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    if binsize > 0:
                        t_err = t_err[mask]
                    
                plot_color = color_dict.get((ins, band), 'black')
                if ins=='NEOWISE':
                    ax.plot(t_observed - ref_mjd, y_observed, linestyle='--', color=plot_color, alpha=0.5)
                ax.errorbar(t_observed - ref_mjd, y_observed, xerr=t_err, yerr=uncert, color=plot_color, label=band, 
                            fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Magnitude", family="serif", fontsize=16)
                    
            if y_ax == 'flux':
                sigma = remove_outliers(file['flux'][file['filter'] == band], method=outliers)
                    
                t_observed = np.array(file["bjd"][file['filter'] == band])[sigma]
                y_observed = np.array(file['flux'][file['filter'] == band])[sigma]
                uncert = np.array(file["fluxerr"][file['filter'] == band])[sigma]
                
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    
                plot_color = color_dict.get((ins, band), 'black')
                ax.errorbar(t_observed - ref_mjd, y_observed, yerr=uncert, color=plot_color, label=band, 
                            fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Flux", family="serif", fontsize=16)
                    
        ax.set_title(f"{source_name}", weight="bold") 
        ax.set_xlabel(f"MJD - {ref_mjd} [days]", family="serif", fontsize=16)
        ax.tick_params(which='major', width=2, direction='out')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, direction='out')
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
        ax.invert_yaxis()
        xmin, xmax = ax.get_xlim()    
        
        def xconv(x):
            tyear = Time(x + ref_mjd, format="mjd")
            return tyear.jyear 
        
        xmin2 = xconv(xmin)
        xmax2 = xconv(xmax)
        
        ay2 = ax.twiny()
        ay2.minorticks_on()
        ay2.tick_params(which='major', width=2, direction='out')
        ay2.tick_params(which='major', length=7)
        ay2.tick_params(which='minor', length=4, direction='out')
        ay2.tick_params(labelsize=16)
        ay2.set_xlim([xmin2, xmax2])
        ay2.set_xlabel('Year', fontsize=16)
        ay2.ticklabel_format(useOffset=False)
            
        colors_for_filters = [color_dict.get((ins, filt), None) for filt in available_filters]
                
        ax.legend(loc="upper right", ncols=len(available_filters), labelcolor=colors_for_filters, 
                  shadow=True, columnspacing=0.8, handletextpad=0.5, handlelength=1, markerscale=1.5, 
                  prop={"weight": "bold"})
        
        if savefig:
            plt.savefig(f'{out_path}/{source_name}.png', bbox_inches="tight", format="png")
        
        if plot:
            plt.show()
        
        if ax is None:
            plt.close()
        
    else:
        print('Less than 5 points in the time series for source '+str(file['name'][0]))
        

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def vel_period_mass(m1, q, P, t_scale='days', e=0, plot=True):
    '''
    Heatmap with possible radial velocities of primary and secondary stars in
    a stellar binary, given a range of masses for the primary and a range for
    the mass ratio, a period of variaton (of the eclispes), and the eccentricity
    of the orbit.
    
    Parameters
    ----------
    m1: array of floats
        Array with mass values for the primary star
    q: array of floats
        Array with values for the mass ratio. Should be (0,1], with steps
        matching the range for m1
    P: float
        Period of variation of the binary eclipses
    t_scale: string, optional
        Timescale in which the period is given. Can be 'days', 'h' (hours), 
        'min' (minutes) or 's' (seconds).
    e: float, optional
        Eccentricity of the orbit.
    plot: bool, optional
        If True, the plot is shown.

    Returns
    -------
    fig: matplotlib figure object
        The generated figure, to save outside the function for example.

    '''
    
    # Set the period in seconds ()
    if t_scale=='days':
        P_s = P*24*3600
    elif t_scale=='h':
        P_s = P*3600
    elif t_scale=='min':
        P_s = P*60
    else:
        if t_scale=='s':
            print('Timescale already in seconds.')
        else:
            print('Unsupported timescale. Use "days", "h", "min" or "s".')
    
    
    # Calculate radial velocities given a range o the primary mass and a period
    def vel_rad(m1, q, P_s, e):
        sin_i = 2/3 # inclination angle is generally unknown, so an integral average with some correcction is used (Pettini L4)
        G = 4.302*10**(-3)*3.086*10**(13) # Grav. const. in km^3 (M_sun)^(-1) s^(-2)
        K_1 = ((q*m1*sin_i*2*np.pi*G)/(P_s*(1-e**2)**(3/2)*(1+(1/q))**2))**(1/3) # In km/s
        K_2 = K_1/q
        return K_1, K_2
    
    m1_grid, q_grid = np.meshgrid(m1, q, indexing='ij')
    K_1, K_2 = vel_rad(m1_grid, q_grid, P_s, e)
    q = np.around(q, decimals=2)
    m1 = np.around(m1, decimals=2)
    min1, max1 = np.around(np.amin(K_1), decimals=2), np.around(np.amax(K_1), decimals=2)
    min2, max2 = np.around(np.amin(K_2), decimals=2), np.around(np.amax(K_2), decimals=2)
    
    
    # # For the primary
    # data_1 = pd.DataFrame({'q=M2/M1': q, 'M1': m1, 'vR1': K_1})
    # data_pivoted_1 = data_1.pivot(columns='q=M2/M1', index='M1', values='vR1')
    
    # # For the secondary
    # data_2 = pd.DataFrame({'q=M2/M1': q, 'M2': q*m1, 'vR2': K_2})
    # data_pivoted_2 = data_2.pivot(columns='q=M2/M1', index='M2', values='vR2')
    
    # Plot
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # For the primary
    cbar_kws={'label': '$v_{r1} (km/s)$', 'extend' : 'both'} 
    sns.heatmap(K_1, ax=ax1, xticklabels=q, yticklabels=m1,
                cmap='inferno', cbar_kws=cbar_kws)
    ax1.set_title(f'$v_r$ of the primary (min={min1}, max={max1} km/s)', 
                  fontsize='x-large')
    ax1.invert_yaxis()
    ax1.set_xlabel('q=M2/M1', fontsize='x-large')
    ax1.set_ylabel('M1', fontsize='x-large')
    ax1.figure.axes[-1].yaxis.label.set_size('x-large')
    
    # For the secondary
    cbar_kws={'label': '$v_{r2} (km/s)$', 'extend' : 'both'}
    sns.heatmap(K_2, ax=ax2, xticklabels=q, yticklabels=m1, 
                cmap='inferno', cbar_kws=cbar_kws)
    ax2.set_title(f'$v_r$ of the secondary (min={min2}, max={max2} km/s)', 
                  fontsize='x-large')
    ax2.invert_yaxis()
    ax2.set_xlabel('q=M2/M1', fontsize='x-large')
    ax2.set_ylabel('M1', fontsize='x-large')
    ax2.figure.axes[-1].yaxis.label.set_size('x-large')
    
    fig.suptitle(f'Period: {P} {t_scale}', fontsize='xx-large')
    if plot==True:
        plt.show()
    
    return fig
    

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def lc_folding(name, time, y, uncert, best_freq, ax, t_start=None, cycles=1, outliers='eb', yunits='mag'):
    '''
    Folds given light curve at a certain frequency.
    
    Parameters
    ----------
    name : string
        Name of the source (e.g. 'Gaia DR3 123')
    time : array of floats
        Array with the time values.
    y : array of floats
        Array with the magnitude values.
    uncert : array of floats
        Array with the magnitude errors.
    best_freq : float
        Value of the frequency to fold the light curve at.
    ax : matplotlib axes
        Where to plot the folded light curve
    t_start : float, optional
        Initial time to fold the light curve. If None, it is set to the time
        with min(y), Default is None.
    cycle : float, optional
        How many phases are shown. Default is 1.
    Returns
    -------
    None

    '''
    
    # Remove outliers
    sigma = remove_outliers(y, method=outliers)
    
    time = np.array(time)[sigma]
    y = np.array(y)[sigma]
    if uncert is not None:
        uncert = np.array(uncert)[sigma]
    
    # Set initial time
    if t_start is None:
        min_index = np.argmax(y)
        time = time - time.iloc[min_index]
    else:
        time = time - t_start
    
    name = name.split('_')[0]
    
    # Plot folded light curve
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.errorbar((time * best_freq) % cycles, y,yerr=uncert, fmt='o', ecolor='black', capsize=2, elinewidth=2,
                 markersize=2, markeredgewidth=0.5, alpha=0.8, color='black',zorder=0, mec='black')
    ax.set_xlabel('Phase', fontsize=19)
    if yunits=='mag':
        ax.invert_yaxis()
        ax.set_ylabel(r'Magnitude', fontsize=19)
    elif yunits=='flux':
        ax.set_ylabel(r'Flux', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # plt.suptitle(name, fontsize=20, weight='bold')
    if ((24*60)/best_freq)<60:
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {(24*60) / best_freq:.3f}' + r'$ \, min$',
                     fontsize=20, weight='bold')
    elif ((((24*60)/best_freq)>60) and (((24*60)/best_freq)<(24*60))):
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {24 / best_freq:.3f}' + r'$ \, hr$',
                     fontsize=20, weight='bold')
    else:
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {1/best_freq:.3f}' + r'$ \, d$',
                     fontsize=20, weight='bold')
   


def lc_combined(name, t_list, y_list, y_list_err, filt_list, best_freq, t_start=None, cycles=2, outliers='eb'):
    '''
    Stacks plots of folded ligth curves of the same source with different 
    instruments and/or filters.

    Parameters
    ----------
    name : string
        Name of the source.
    t_list : list with arrays of floats
        Each array in the list contains the time axis of the ligth curve in a
        different instrument and/or filter.
    y_list : list with arrays of floats
        Each array in the list contains the y axis of the ligth curve in a
        different instrument and/or filter.
    y_list_err : list with arrays of floats
        Each array in the list contains the errors of the y axis of the ligth 
        curve in a different instrument and/or filter.
    filt_list : list of strings
        List of names of the different instruments/filters.
    best_freq : float
        Value of the frequency to fold the light curve at.
    t_start : float, optional
        Initial time to fold the light curve. If None, it is set to the time
        with min(y), Default is None.

    Returns
    -------
    None.

    '''
    
    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF, zg'):'g',
        ('ZTF, zr'):'r',
        ('ZTF, zi'):'goldenrod',
        ('IRSA_ZTF, zg'):'g',
        ('IRSA_ZTF, zr'):'r',
        ('IRSA_ZTF, zi'):'goldenrod',
        ('ASAS-SN, V'):'darkcyan',
        ('ASAS-SN, g'):'blue',
        ('ATLAS, o'):'orange',
        ('ATLAS, c'):'cyan',
        ('NEOWISE, W1'):'darkred',
        ('NEOWISE, W2'):'slategray',
        ('BLACKGEM, u'):'purple',
        ('BLACKGEM, g'):'skyblue',
        ('BLACKGEM, r'):'orange',
        ('BLACKGEM, i'):'firebrick',
        ('BLACKGEM, z'):'sienna',
        ('BLACKGEM, q'):'black',
        ('TESS, QLP'):'magenta',
        ('TESS, SPOC'):'magenta',
        ('TESS, TASOC'):'magenta',
        ('TESS, CDIPS'):'magenta',
        ('TESS, TGLC'):'magenta',
        ('TESS, GSFC-ELEANOR-LITE'):'magenta'}
    
    fig = plt.figure(figsize=(10,4*len(t_list)))
    height_ratios = np.repeat(1, len(t_list))
    gs = gridspec.GridSpec(len(t_list), 1, height_ratios=height_ratios)
    
    for i in range(len(t_list)):
        if filt_list[i].split(',')[0]=='TESS':
            yunits='flux'
        else:
            yunits='mag'
            
        if i==0:
            ax1 = fig.add_subplot(gs[i])
            lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax1, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
            # Change color depending on ins and filter
            errorbars = ax1.get_children()
            plot_color = color_dict.get(filt_list[i], 'black')
            for j in range(4):
                errorbars[j].set_color(plot_color)
            ax1.tick_params(axis='x', labelbottom=False, direction='in')
            ax1.set_xlabel('')
            trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
            ax1.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
        elif i==max(range(len(t_list))):
            ax = fig.add_subplot(gs[i], sharex=ax1)
            lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
            # Change color depending on ins and filter
            errorbars = ax.get_children()
            plot_color = color_dict.get(filt_list[i], 'black')
            for j in range(4):
                errorbars[j].set_color(plot_color)
            if filt_list[i].split(',')[0]=='TESS':
                errorbars[1].remove()
                errorbars[2].remove()
                errorbars[3].remove()
            ax.set_title('')
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
            ax.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
        else:
            ax = fig.add_subplot(gs[i], sharex=ax1)
            lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
            # Change color depending on ins and filter
            errorbars = ax.get_children()
            plot_color = color_dict.get(filt_list[i], 'black')
            for j in range(4):
                errorbars[j].set_color(plot_color)
            ax.set_title('')
            ax.tick_params(axis='x', labelbottom=False, direction='in')
            ax.set_xlabel('')
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
            ax.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
        
    gs.update(hspace=0)
    
    
def bulk_combine(name, instruments, best_freq, cycles=2, outliers='eb'):
    '''
    Calls lc_combined with the instruments given.
    
    Parameters
    ----------
    name : string
        Name of the source.
    instruments : list of strings
        Instruments from which the data is taken from. Available now are: 
            ZTF, IRSA_ZTF, ATLAS, ASAS-SN, NEOWISE, BLACKGEM, TESS.
    best_freq : float
        Value of the frequency to fold the light curve at..

    Returns
    -------
    None.

    '''
    times=[]
    mags=[]
    mag_errs=[]
    filts=[]
    for ins in instruments:
        if ins!='TESS':
            nams = name.split('_')[0]
            table = pd.read_csv(f'{ins}_lightcurves_std/{nams}.csv')
        else:
            table = pd.read_csv(f'{ins}_lightcurves_std/{name}.csv')
        bands=list(set(table['filter']))
        if 'ZTF' in ins:
            desired_order = ['zg', 'zr', 'zi']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
        elif 'ATLAS' in ins:
            desired_order = ['c', 'o']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
        elif 'ASAS' in ins:
            desired_order = ['g', 'V']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
            
        for i, band in enumerate(bands):
            t=table['bjd'].loc[table['filter']==band]
            if ins=='TESS':
                y=table['flux'].loc[table['filter']==band]
                yerr=table['fluxerr'].loc[table['filter']==band]
            else:
                y=table['mag'].loc[table['filter']==band]
                yerr=table['magerr'].loc[table['filter']==band]
            if i == 0:
                min_index = np.argmax(y)
                t_start = t.iloc[min_index]
            filt = table['inst'][0] + ', ' + band
            times.append(t)
            mags.append(y)
            mag_errs.append(yerr)
            filts.append(filt)

    lc_combined(name, times, mags, mag_errs, filts, best_freq, t_start=t_start, cycles=cycles, outliers=outliers)

    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def lomb_scargle(t, y, yerr=None, fmin=None, fmax=None, plot=True):
    '''
    Uses Lomb-Scargle to find the best frequency of the given time-series.
    
    Parameters
    ----------
    t: array of floats
        Array with the time values.
    y: array of floats
        Array with the magnitude values.
    yerr: array of floats, optional
        Array with the magnitude errors.
    fmin, fmax: float, optional
        Values of the minimum and maximum frequencies (respectively) of the
        search grid.
    plot: boolean, optional
        If set to True, shows the plot of the Lomb-Scargle periodogram and the
        folded light curve at the best frequency found.

    Returns
    -------
    frequency: array of floats
        Array with the frequency values in the search grid.
    power:
        Array with the power value of each frequency in the search grid.
    fig: matplotlib figure object
        The generated figure, to save outside the function for example.

    '''
    
    # Remove outliers
    sigma = remove_outliers(y, method='eb')
    
    # Variables to input into the FINKER script
    t = np.array(t)[sigma]
    y = np.array(y)[sigma]
    yerr = np.array(yerr)[sigma]
    
    # Set units
    y = y * u.mag
    yerr = yerr * u.mag
    t = t * u.day
    if fmin is not None:
        fmin, fmax = fmin/u.day, fmax/u.day
    
    frequency, power = LombScargle(t, y, yerr).autopower(minimum_frequency=fmin,
                                                         maximum_frequency=fmax)

    best_freq = frequency[np.argmax(power)]
    print(f'Best frequency: {best_freq}')
    period = 1/best_freq
    if (period.value<1) and (period.value>1/24):
        period = period.to(u.h)
    elif (period.value<1/24) and (period.value>1/(24*60)):
        period = period.to(u.min)
    elif period.value<1/(24*60):
        period = period.to(u.s)
    
    
    # Set initial time
    max_index = np.argmax(y)
    t = t - t[max_index]
    
    if plot==True:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        ax1.plot(frequency, power)
        ax1.set_title('Lomb-Scargle Periodogram', fontsize=18)
        ax1.set_xlabel('Frequency (1/d)', fontsize=18)
        ax1.set_ylabel('Power', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        ax2.errorbar((t * best_freq) % 1, y,yerr=yerr, fmt='o', ecolor='black', capsize=2, elinewidth=2,
                     markersize=2, markeredgewidth=2, alpha=0.8, color='black',zorder=0)
        ax2.set_xlabel('Phase', fontsize=18)
        ax2.set_ylabel(r'Magnitude', fontsize=18)
        ax2.invert_yaxis()
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_title(f'Frequency: {best_freq.value:.4f} ({best_freq.unit}); period: {period.value:.4f} ({period.unit})', 
                      fontsize=18)
        
        
        plt.show()
        
    return frequency, power, fig

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def next_transits(name, observing_time, eclipse_time, orbital_period, eclipse_duration=None, 
                  eclipse_time_format='mjd', n_eclipses=5, verbose=True):
    '''
    Gives the next n_eclipses transits of an object with an eclipse at
    eclipse_time and an orbital_period, starting at observing_time.

    Parameters
    ----------
    name : string
        Name of the object to observe.
    observing_time : string or astropy.Time object
        Date ('yyyy-mm-dd hh-mm-ss') from which dates of transit are computed.
    eclipse_time : string or astropy.Time object
        Date ('yyyy-mm-dd hh-mm-ss') when a (primary) eclipse happened.
    orbital_period : float with units
        Period of the eclipse, with units (astropy.units).
    eclipse_duration : float with units, optional
        Duration of the eclipse. The default is None.
    eclipse_time_format : string, optional
        Format of the elcipse_time. The default is 'mjd'.
    n_eclipses : int, optional
        Number of next transits computed. The default is 5.
    verbose : bool, optional
        If True, text with the calculations is shown. If False, the function
        does the calculations and returns the dates of the next primary and 
        secondary eclipses. The default is True.

    Returns
    -------
    next_pe : list of dates
        Dates of the next primary eclipses.
    next_se : list of dates
        Dates of the next secondary eclipses.

    '''
    observing_time = Time(observing_time)
    primary_eclipse_time = Time(eclipse_time, format=eclipse_time_format)
    system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                           orbital_period=orbital_period, duration=eclipse_duration,
                           name=name)
    next_pe = system.next_primary_eclipse_time(observing_time, n_eclipses=n_eclipses)
    next_se = system.next_secondary_eclipse_time(observing_time, n_eclipses=n_eclipses)
    
    if verbose:
        print(f'| Observing "{name}" at {observing_time} |')
        print('-----------------------------------------------------------------')
        print(f'Next {n_eclipses} primary eclipses:')
        print(next_pe)
        print('-----------------------------------------------------------------')
        print(f'Next {n_eclipses} secondary eclipses:')
        print(next_se)
        print('-----------------------------------------------------------------')
            
        if eclipse_duration is not None:
            print(f'Next {n_eclipses} primary ingress and egress times:')
            ie_t = system.next_primary_ingress_egress_time(observing_time, n_eclipses=n_eclipses)
            ie_t.format='iso'
            print(ie_t)
            print('-----------------------------------------------------------------')
            print(f'Next {n_eclipses} secondary ingress and egress times:')
            ie_t = system.next_secondary_ingress_egress_time(observing_time, n_eclipses=n_eclipses)
            ie_t.format='iso'
            print(ie_t)
            
    return next_pe, next_se
        

        
def observable(name, ra, dec, site_name, time_range, obs_date=None,
               phase_range=None, event_time=None, period=None, eclipse_time_format='mjd',
               twilight='astronomical', min_altitude=30*u.deg):
    '''
    Tells if an object is observable or not given some conditions.'
    There are three options:
        - If phase_range is set: observability of an object with some
        periodicity given the conditions, and the object is in a phase between
        the ones given in phase_range.
        - If phase_range is None, and obs_date is set: observability of an
        object in specific dates given in obs_date. Basically, is the object
        observable at ['yyyy-mm-dd hh-mm-ss', ('yyyy-mm-dd hh-mm-ss'), ...]?
        - If phase_range and obs_date are None: observability of an object in
        a wide range of dates. Basically, is the object observable in any nigth
        between 'yyyy-mm-dd hh-mm-ss' and 'yyyy-mm-dd hh-mm-ss'? (These dates
        are the ones specified in time_range).

    Parameters
    ----------
    name : string
        Name of the object to observe.
    ra : float
        Right ascension of the object to observe, in degrees.
    dec : float
        Deeclination of the object to observe, in degrees.
    site_name : string
        Name of the site of the observation. Needs to be a name in the astropy-
        -data repository
    time_range : tuple
        Can be a tuple of floats indicating the hours to start and end
        observations when obs_date is set (for specific observation dates).
        A tuple of dates ('yyyy-mm-dd hh-mm-ss') to check observability in a 
        wider range, or if phase_range is set.
    obs_date : list of dates, optional
        Specific dates to observe. Can be convined with next_transits to check
        if the object is observable during the transits. The default is None.
    phase_range : tuple, optional
        Check if the object is observable while in a phase between the two 
        values in the phase_range. The default is None.
    event_time : string or astropy.Time object, optional
        Date ('yyyy-mm-dd hh-mm-ss') when a an event happened. Only relevant if
        phase_range is set. The default is None.
    period : float with units, optional
        Period of the eclipse, with units (astropy.units). Only relevant if
        phase_range is set. The default is None.
    eclipse_time_format : string, optional
        Format of the elcipse_time.Only relevant if phase_range is set. The 
        default is 'mjd'.
    twilight : string, optional
        Either 'astronomical', 'nautical' or 'civil'. The default is
        'astronomical'.
    min_altitude : float with units, optional
        Constrain on the minimum altitude accepted. The default are 30 degrees.

    Raises
    ------
    Exception
        An exception is raised if the phase_range is set, but no event_time and
        no period are given for the periodic event. Also, if the twilight is not
        given correctly.

    Returns
    -------
    None.

    '''
    print('True means that the object is observable with the conditions given.')
    
    # Set twilight
    if twilight == 'astronomical':
        twilight = AtNightConstraint.twilight_astronomical()
    elif twilight == 'nautical':
        twilight = AtNightConstraint.twilight_nautical()
    elif twilight == 'civil':
        twilight = AtNightConstraint.twilight_civil()
    else:
        raise Exception('Twilight must be one of: "astronomical", "nautical" or "civil"')
    
    # Target to observe and site of observation
    target = FixedTarget(SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')) 
    site = Observer.at_site(site_name)  # Site of observations
    
    # Check observability in a given phase range
    if phase_range is not None:
        if event_time is None or period is None:
            raise Exception('To see if a phase range is observable, seting event_time and period is needed.')
        epoch = Time(event_time, format=eclipse_time_format)  # Reference time of periodic event  
        event = PeriodicEvent(epoch=epoch, period=period)
        start_time = Time(time_range[0])  # Start of observing time
        end_time = Time(time_range[1])  # End of observing time
        constraints = [PhaseConstraint(event, min=phase_range[0], max=phase_range[1]),
                       twilight, AltitudeConstraint(min=min_altitude)]
        print(is_observable(constraints, site, target, time_range=[start_time, end_time]))
    
    else:
        # Check observability in specific times (e.g. '2024-04-12 04:23:11', '2024-04-13 01:45:59')
        if obs_date is not None:
            min_t = dt.time(time_range[0])
            max_t = dt.time(time_range[1])
            constraints = [twilight, AltitudeConstraint(min=min_altitude),
                           LocalTimeConstraint(min=min_t, max=max_t)]
            print(is_event_observable(constraints, site, target, times=obs_date))
        # Check observability in a range of dates (e.g. any night from '2024-04-12 20:00:00' to '2024-04-21 06:00:00') 
        else:
            start_time = Time(time_range[0])  # Start date
            end_time = Time(time_range[1])  # Ending date
            constraints = [twilight, AltitudeConstraint(min=min_altitude)]
            print(is_observable(constraints, site, target, time_range=[start_time, end_time]))
        
        
        
def observable_when(name, ra, dec, site_name, time_range, delta, phase_range=None, event_time=None, period=None, 
                    eclipse_time_format='mjd', twilight='astronomical', min_altitude=30*u.deg, verbose=False):
    '''
    Checks if an object is observable during a time period, checking every
    time step (delta).

    Parameters
    ----------
    name : string
        Name of the object to observe.
    ra : float
        Right ascension of the object to observe, in degrees.
    dec : float
        Deeclination of the object to observe, in degrees.
    site_name : string
        Name of the site of the observation. Needs to be a name in the astropy-
        -data repository
    time_range : tuple
        A tuple with the starting and ending dates ('yyyy-mm-dd hh-mm-ss') to
        check observability.
    delta : float with units
        Time step.
    phase_range : tuple, optional
        Check if the object is observable while in a phase between the two 
        values in the phase_range. The default is None.
    event_time : string or astropy.Time object, optional
        Date ('yyyy-mm-dd hh-mm-ss') when a an event happened. Only relevant if
        phase_range is set. The default is None.
    period : float with units, optional
        Period of the eclipse, with units (astropy.units). Only relevant if
        phase_range is set. The default is None.
    eclipse_time_format : string, optional
        Format of the elcipse_time.Only relevant if phase_range is set. The 
        default is 'mjd'.
    twilight : string, optional
        Either 'astronomical', 'nautical' or 'civil'. The default is
        'astronomical'.
    min_altitude : float with units, optional
        Constrain on the minimum altitude accepted. The default are 30 degrees.
    verbose : bool, optional
        If True, prints some stuff to see what the function is doing. The 
        default is False.

    Raises
    ------
    Exception
        An exception is raised if the twilight is not given correctly.

    Returns
    -------
    observable_dates : list
        List with the dates when the target is observable.

    '''
    # Set twilight
    if twilight == 'astronomical':
        twilight = AtNightConstraint.twilight_astronomical()
    elif twilight == 'nautical':
        twilight = AtNightConstraint.twilight_nautical()
    elif twilight == 'civil':
        twilight = AtNightConstraint.twilight_civil()
    else:
        raise Exception('Twilight must be one of: "astronomical", "nautical" or "civil"')
        
    # Target to observe and site of observation
    target = FixedTarget(SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')) 
    site = Observer.at_site(site_name)  # Site of observations
    
    start_time = Time(time_range[0])  # Start of observing time
    end_time = Time(time_range[1])  # End of observing time
    dt = end_time-start_time  # TimeDelta when observations are desired
    delta = TimeDelta(delta)
    n_steps = int(dt/delta)  # Number of dates/times to check. 1+ to check both starting and ending dates
    date = start_time  # Setup the starting date/time
    check_obs_dates = [date]  # List of dates to check observability
    # Fill the list with the dates that will be checked for observability
    for _ in range(n_steps):
        date += delta
        check_obs_dates.append(date)
    if verbose:
        print('Number of steps: ', n_steps)
        print('Checking dates:')
        for t in check_obs_dates:
            print(t.value)
        print('Observable?:')
    observable_dates=[]  # List of dates the object will be observable with the desired conditions
    # Check observability in a given phase range
    if phase_range is not None:
        if event_time is None or period is None:
            raise Exception('To see if a phase range is observable, seting event_time and period is needed.')
        epoch = Time(event_time, format=eclipse_time_format)  # Reference time of periodic event  
        event = PeriodicEvent(epoch=epoch, period=period)
        constraints = [PhaseConstraint(event, min=phase_range[0], max=phase_range[1]),
                       twilight, AltitudeConstraint(min=min_altitude)]
        for time, next_time in zip(check_obs_dates, check_obs_dates[1:]):
            check = is_observable(constraints, site, target, time_range=[time, next_time])[0]
            if verbose:
                print([time.value, next_time.value], check)
            if check is True:
                observable_dates.append([time.value, next_time.value])
    else:
        constraints = [twilight, AltitudeConstraint(min=min_altitude)]
        for time, next_time in zip(check_obs_dates, check_obs_dates[1:]):
            check = is_observable(constraints, site, target, time_range=[time, next_time])[0]
            if verbose:
                print([time.value, next_time.value], check)
            if check == True:
                observable_dates.append([time.value, next_time.value])
        
    if len(observable_dates) == 0:
        print('---')
        print(f'The object {name} is not observable between {time_range[0]} and {time_range[1]}.')
    else:
        print('---')
        print(f'The object {name} is obsevable during the following period(s):')
        return observable_dates
    
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def oc_diagram(time, y, period, t_eclipse=None, y_eclipse=None, time_format='mjd'):
    
    
    # Remove outliers from the data
    sigma = remove_outliers(y, method='eb')
    time = time[sigma]
    y = y[sigma]
    
    # Set parameters and units
    time_with_units = Time(time, format=time_format)
    # period = period*u.d
    
    years = np.array([t.datetime.year for t in time_with_units])
    years_unique = list(set(years))
    
    # Set the eclipse time to the time when the maximum magnitude was observed
    if t_eclipse is None:
       t_eclipse = time[np.argmax(y)]
    # Also set the magnitude value at primary mid-eclipse
    if y_eclipse is None:
        y_eclipse = y[np.argmax(y)]
        
    # Set parameters and units
    t_eclipse = Time(t_eclipse, format=time_format)
    mintime = Time(min(time), format=time_format)  # Starting time of the lightcurve
    n_eclipses = int((max(time)-min(time))/period)  # Number of eclipses in the time range given
    
    # Calculated eclipses
    primary_eclipses, secondary_eclipses = next_transits('', mintime, t_eclipse, period*u.d, 
                                                         n_eclipses=n_eclipses, verbose=False)
    
    eclipse_years = np.array([t.datetime.year for t in primary_eclipses])
    
    oc = []
    number_eclipses = []
    for year in years_unique:
        mask = years==year
        t = time[mask]
        mag = y[mask]

        first_eclipse = primary_eclipses[eclipse_years==year][0]
        
        t_start = first_eclipse.mjd - 0.5*period
        t = t - t_start
        x = (t * 1/period) % 1
        
        sorted_indices = np.argsort(x)
        sorted_times = x[sorted_indices]
        sorted_magnitudes = mag[sorted_indices]
        
        
        y_min = max(sorted_magnitudes[(sorted_times>0.4)*(sorted_times<0.6)])
        min_index = np.where(sorted_magnitudes==y_min)[0][0]
        start_index = max(0, min_index - (11 // 2))
        end_index = min(len(sorted_indices)-1, start_index + 11)
        
        # Select the data points within the window
        t_to_fit = sorted_times[start_index:end_index]
        mag_to_fit = sorted_magnitudes[start_index:end_index]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        p = np.polyfit(t_to_fit,mag_to_fit,2)
        time_fit = np.linspace(sorted_times[start_index], sorted_times[end_index], 1000)
        y_fit = np.polyval(p, time_fit)
        oc.append((time_fit[np.argmax(y_fit)] - 0.5)*period)
        ax.axvline(time_fit[np.argmax(y_fit)], color='r', ls='--')
        ax.plot(time_fit, y_fit, ls='--', color='r')
        ax.scatter(t_to_fit, mag_to_fit)
        
        ax.scatter(x, mag, s=10, color = 'k')
        ax.axvline(0.5)
        ax.set_xlabel('Phase', fontsize=19)
        ax.set_ylabel(r'Magnitude', fontsize=19)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.set_xlim(0.4, 0.6)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    
    # # Set the eclipse time to the time when the maximum magnitude was observed
    # if t_eclipse is None:
    #    t_eclipse = time[np.argmax(y)]
    # # Also set the magnitude value at primary mid-eclipse
    # if y_eclipse is None:
    #     y_eclipse = y[np.argmax(y)]
    
    # # Set parameters and units
    # time = Time(time, format=time_format)
    # period = period*u.d
    # mintime = Time(min(time))  # Starting time of the lightcurve
    # t_eclipse = Time(t_eclipse, format=time_format)  
    # n_eclipses = int((max(time)-min(time))/period)  # Number of eclipses in the time range given
    
    # # Calculated eclipses
    # primary_eclipses, secondary_eclipses = next_transits('', mintime, t_eclipse, period, 
    #                                                      n_eclipses=n_eclipses, verbose=False)
        
    # # Slice the light curve, each slice contains one eclipse (primary or secondary)
    # oc = []
    # number_eclipses = []
    # if mode=='fold':
    #     first_eclipse = primary_eclipses[0]
    #     starting_point = first_eclipse-0.5*period
    #     slice_size = 365
        
        
    # else:
    #     first_eclipse = primary_eclipses[0]
    #     starting_point = first_eclipse - 0.25*period # We set this as a kind of phase=0, so each eclipse falls in the middle of the slice more or less
    #     slice_size = period*0.5
        
    #     for i in range(len(primary_eclipses)):
    #         slice_start = starting_point + i*period
    #         slice_end = slice_start + slice_size
    #         mask = (slice_start <= time)*(time <= slice_end)
    #         if any(mask)==True:
    #             y_min = max(y[mask])
    #             min_index = np.where(y==y_min)[0][0]
    #         else:
    #             continue
    #         # primary eclipse
    #         #print(y[min_index])
    #         if (y_eclipse-0.03 < y[min_index]) and (y[min_index] < y_eclipse+0.03):
    #             y_to_fit = [y[min_index-2], y[min_index-1], y[min_index], 
    #                         y[min_index+1], y[min_index+2]]
    #             time_to_fit = [time[min_index-2].value, time[min_index-1].value, time[min_index].value, 
    #                         time[min_index+1].value, time[min_index+2].value]
    #             p = np.polyfit(time_to_fit,y_to_fit,2)
    #             time_fit = np.linspace(time[min_index-2].value, time[min_index+2].value, 1000)
    #             y_fit = np.polyval(p, time_fit)
    #             oc.append(time_fit[np.argmax(y_fit)]-primary_eclipses[i].value)
    #             number_eclipses.append(i)
                
    #             plt.figure(figsize=(8,8))
    #             plt.scatter(time[mask].value, y[mask], s=5, c='b')
    #             plt.plot(time_fit, y_fit, ls='--', color='r')
    #             ax = plt.gca()
    #             ax.invert_yaxis()
    #             plt.show()
    #             plt.close()
    
    #obs_secondary_eclipses = []
    
    
    # sigma = remove_outliers(oc)
    plt.figure(figsize=(8,8))
    plt.scatter(years_unique, np.array(oc), s=10, c='black')
    # plt.scatter(number_eclipses, oc, s=10, c='black')
    plt.xlabel('Eclipse')
    plt.ylabel('O-C (days)')
    plt.show()
    plt.close()
    
    return number_eclipses, oc  
    