#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:02:52 2023

@author: nadiablago
"""

from ztfquery import lightcurve
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.table import Table
import os, glob
from astropy.time import Time
from pyasassn.client import SkyPatrolClient

path= "./ztf_lightcurves"

if not os.path.isdir(path):
    os.makedirs(path)

#---------------------------------------------------------------------------------------------------------------#
#ZTF
#---------------------------------------------------------------------------------------------------------------#
def query_lc_ztf(ra, dec, savefile, searchrad=2, plot=False):
    '''
    Uses de ztfquery interface to query all detections within a search radius of a given
    ra and dec.
    Stores results into a file.

    Parameters
    ----------
    ra : double
        right ascension.
    dec : double
        declination.
    savefile : string, optional
        Name of the file where the lightcurve will be stored in csv format.
    searchrad : double, optional
        Search radius in arcseconds. The default is 2.
    plot : boolean, optional
        Set to True to plot the lightcurve
    Returns
    -------
    None.

    '''
    
    lcq = lightcurve.LCQuery.from_position(ra, dec, 2)
    
    lc = Table.from_pandas(lcq.data)
    
    #Remove flagged exposures to emilinate bad datapoints.
    lc = lc[lc['catflags'] == 0]
    
    if len(lc) > 0:
        lc.write(savefile, format="ascii.csv", overwrite=True)
    
    if plot:
        #Method from ztfquery
        lcq.show()
        
        #Custom method
        '''dates = time.Time(np.asarray(lc["mjd"], dtype="float"), format="mjd").datetime
        plt.errorbar(dates, lc['mag'], yerr=lc['magerr'], fmt="bo")
        
        plt.gca().invert_yaxis()'''
        

    
    
def download_ztf_lightcurves(asciicords, name):
    '''
    
    Downloads the lightcurves for all the coordinates containted in the file
    asciicords.
    
    Parameters
    ----------
    asciicords : string
        Path of the file in CSV format where the coordinates of the sources are.
        The file must contain at least 3 columns: name, ra, dec

    Returns
    -------
    None.
    
    Saves the files with the name of the source in the defined path.

    '''
    t = Table.read(asciicords, format="ascii.csv")
    for i, ti in enumerate(t):
        if i%10 ==0:
            print ("processing ",i)
        query_lc_ztf(ti["ra"], ti["dec"], os.path.join(path, "%s.csv"%ti[name]), 2)
    

#---------------------------------------------------------------------------------------------------------------#
#ASAS_SN
#---------------------------------------------------------------------------------------------------------------#
def download_asas_sn_lightcurves(asciicords, name):
    '''
    Downloads the lightcurves for all the coordinates containted in the file
    asciicords, and saves the plotted lightcurves.
    
    Parameters
    ----------
    asciicords : string
        Path of the file in CSV format where the coordinates of the sources are.
        The file must contain at least 3 columns: name, ra, dec.
    name: string
        Name of the column with the IDs of the sources (e.g. Gaia source ID).

    Returns
    -------
    None.

    '''
    table = pd.read_csv(asciicords)
    
    client = SkyPatrolClient()
    
    path= "./asas_sn_lightcurves"
    
    if not os.path.isdir(path):
        os.makedirs(path)
    
    for i in range(len(table)):
        ra = table['ra'][i]
        dec = table['dec'][i]
            
        cone = client.cone_search(ra_deg=ra, 
                                  dec_deg=dec,
                                  radius=5/3600, 
                                  catalog='master_list')
            
        if len(cone)>0:
            lcs = client.cone_search(ra_deg=ra, 
                                     dec_deg=dec, 
                                     radius=5/3600, 
                                     catalog='master_list',
                                     download=True)
                
            asas_sn_id = lcs.stats().index[0]
            lightcurve = lcs[asas_sn_id]
            data = lightcurve.data
            data.to_csv(f'{path}/{table[name][i]}.csv')
