# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:13:20 2024

@author: Gerard Garcia
"""

import os
from astropy.table import Table
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phot_utils # my own
import seaborn as sns
from astropy.timeseries import LombScargle
import astropy.units as u


def standard_table(ins, lc_directory, asciicords, source_name, output_format='csv', suffix='csv'):
    '''
    Given data from an specified catalog/instrument, creates a table with 
    relevant data for plotting light curves, with columns:
        - inst: the catalog/instrument from which the data is taken
        - filter: the band used by the instrument
        - mjd: the Modified Julian Date of the observation
        - mjderr: the error on the mjd
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
    source_name: string
        Name of the column containing the name/id of the sources (to give
        names to the plots and the output tables).
    output_format: string, optional
        The format of the output tables (csv, latex...).
    suffix: string, optional
        Suffix of the output files (csv, tex...).

    Returns
    -------
    None

    '''
    
    # Format for the output table
    mydtype=[("name", "|S25"), ("inst","|S15"), ("filter","|S15"), ("mjd","<f4"),("mjderr","<f4"), 
             ("mag","<f4"), ("magerr","<f4"), ("flux","<f4"), ("fluxerr","<f4")]
    
    # Read directory with light curve files
    if ins != 'NEOWISE':
        lc = os.listdir(lc_directory)
    
    # Read the file with the ids and coordinates of the sources
    if ins != 'BLACKGEM':
        table_coord = pd.read_csv(asciicords)
    
    
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
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['filtercode']
            final_table['mjd'] = table_clean['mjd']
            final_table['mag'] = table_clean['mag']
            final_table['magerr'] = table_clean['magerr']
            
            final_table = Table(final_table)
            
            # Give a name to each file acording to the asciicords file which has the id and the coordinates of the sources
            ra = round(table_clean['ra'][0], 2)
            dec = round(table_clean['dec'][0], 2)
            file_name = table_coord[source_name][(round(table_coord['ra'],2)==ra) & (round(table_coord['dec'],2)==dec)].iloc[0]
            final_table['name'] = np.repeat(file_name, len(final_table))
            
            # Write the output table in the desired directory
            final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
            
    
    # Light curves from All-Sky Automated Surveey for Supernovae [ASAS-SN] (https://asas-sn.osu.edu)   
    if ins == 'ASAS_SN':
          # Create directory where to store the output files
          output_path = './ASAS_SN_lightcurves_std'
          if not os.path.isdir(output_path):
              os.makedirs(output_path)
          
          # Loop for every light curve file in the lc directory
          for file in lc:
              # Read light curve file
              table = Table.read(f'{lc_directory}/{file}', format='ascii.csv')
              
              # Clean the data (bad observations have mag_err = 99.99; quality flag can be G (good) and B (bad))
              table_clean = table[(table['mag_err']<99) & (table['quality']=='G')]
              
              # Create and fill output table
              final_table = np.zeros(len(table_clean), dtype=mydtype)
              
              final_table['inst'] = ins
              final_table['filter'] = table_clean['phot_filter']
              final_table['mjd'] = Time(table_clean['jd'], format='jd').mjd
              final_table['mag'] = table_clean['mag']
              final_table['magerr'] = table_clean['mag_err']
              final_table['flux'] = table_clean['flux']
              final_table['fluxerr'] = table_clean['flux_err']
              
              final_table = Table(final_table)
              
              # Name of the file, the same as the imput lc file because the downloaded tables from asas-sn don't have coordinates
              file_name_only = file.split('.csv')[0]
              final_table['name'] = np.repeat(file_name_only, len(final_table))
              
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
            
            # Create and fill output table
            final_table = np.zeros(len(table_clean), dtype=mydtype)
            
            final_table['inst'] = ins
            final_table['filter'] = table_clean['F']
            final_table['mjd'] = table_clean['##MJD']
            #final_table['mjderr'] = 
            final_table['mag'] = table_clean['m']
            final_table['magerr'] = table_clean['dm']
            final_table['flux'] = table_clean['uJy']
            final_table['fluxerr'] = table_clean['duJy']
            
            final_table = Table(final_table)
            
            # Give a name to each file acording to the asciicords file which has the id and the coordinates of the sources
            ra = round(table_clean['RA'][0], 2)
            dec = round(table_clean['Dec'][0], 2)
            file_name = table_coord[source_name][(round(table_coord['ra'],2)==ra) & (round(table_coord['dec'],2)==dec)].iloc[0]
            final_table['name'] = np.repeat(file_name, len(final_table))
            
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
        neowise_bulk_file = pd.read_csv(lc_directory) # Read the file with all the sources' light curves
        ra_list = neowise_bulk_file['ra_01'].unique() 
        dec_list = neowise_bulk_file['dec_01'].unique()
        if len(ra_list)>1: #Not needed if there's only one source
            for RA,DEC in zip(ra_list,dec_list):
                # Get the name/id of the source for the individual file names
                ra = round(RA, 2)
                dec = round(DEC, 2)
                file_name = table_coord[source_name][(round(table_coord['ra'],2)==ra) & (round(table_coord['dec'],2)==dec)].iloc[0]
                # Create dataframe with an individual light curve and save it
                separated_lc = neowise_bulk_file[neowise_bulk_file['ra_01']==RA]
                separated_lc.to_csv(f'{pathos}/{file_name}.csv', mode='w')
        
        # Loop for every light curve file in the lc directory
        lc = os.listdir(pathos)
        for file in lc:
            # Read light curve file
            table = Table.read(f'{pathos}/{file}', format='ascii.csv')
            
            # Clean the data using cc_flags flag (Contamination and confusion flags) and ph_qual flag (Photometric quality flag)
            # (https://wise2.ipac.caltech.edu/docs/release/neowise/expsup/sec2_1a.html)
            # cc_flags = 0000; ph_qual = A, B or C; [qual_frame > 0 could be used as well]
            # ph_qual_list = ['AA','AB','AC','BA','BB','BC','CA','CB','CC']
            # table_clean = table[(table['cc_flags']=='0000')]
            # ph_qual_mask = np.array([(qual in ph_qual_list) for qual in table_clean['ph_qual']])
            # table_clean = table_clean[ph_qual_mask]
            
            # Clean the data using the signatl to noise ratio and chi^2
            # (https://wise2.ipac.caltech.edu/docs/release/neowise/expsup/sec2_1a.html)
            noise_mask = np.array( (table['w1snr']>2) * (table['w2snr']>2) * 
                                  (table['w1rchi2']<150) * (table['w2rchi2']<150) )
            table_clean = table[noise_mask]
            
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
            
            final_table = Table(final_table)
            
            # Give a name to each file acording to the asciicords file which has the id and the coordinates of the sources
            ra = round(table_clean['ra_01'][0], 2)
            dec = round(table_clean['dec_01'][0], 2)
            file_name = table_coord[source_name][(round(table_coord['ra'],2)==ra) & (round(table_coord['dec'],2)==dec)].iloc[0]
            final_table['name'] = np.repeat(file_name, len(final_table))
            
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
                    final_table['mag'] = table_clean['MAG_OPT']
                    final_table['magerr'] = table_clean['MAGERR_OPT']
                        
                    final_table = Table(final_table)
                    file_name = table_clean['SOURCE_ID'][0]
                        
                    # Write the output table in the desired directory
                    final_table.write(f'{output_path}/{file_name}.{suffix}', format=output_format, overwrite=True)
    

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
        Method used to remove outliers. Either 'median' (removes points a number
        of standard deviations away from the median) or 'iqr' (InterQuartile 
        Range), or None to not remove any outliers.
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
        
    if method==None:
        sigma = np.repeat(True, len(y))
                        
    return sigma


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
    file: string
        File with the light curve data
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
        data points are ploted
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
        print('xdd')
        
        
        
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def plot_ind_lightcurves(file_name, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, rang=None, plot=False, savefig=True):
    '''
    Plots the light curves of the given file. The light curve files MUST
    be in the format given by the function lightcurve_utils.standard_table
    
    Parameters
    ----------
    file: string
        File with the light curve data
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
        data points are ploted
    plot: boolean, optional
        If True, shows the plot
    safefig: boolean, optional
        If True, safe the figure in the directory '{ins}_plots' (e.g. plots for
        ZTF light curves would be stored in 'ZTF_plots')
    
    Returns
    -------
    None

    '''
    
    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF', 'zg'):'g',
        ('ZTF', 'zr'):'r',
        ('ZTF', 'zi'):'gold',
        ('ASAS_SN', 'V'):'darkcyan',
        ('ASAS_SN', 'g'):'blue',
        ('ATLAS', 'o'):'orange',
        ('ATLAS', 'c'):'cyan',
        ('NEOWISE', 'W1'):'darkred',
        ('NEOWISE', 'W2'):'slategray',
        ('BLACKGEM', 'u'):'purple',
        ('BLACKGEM', 'g'):'skyblue',
        ('BLACKGEM', 'r'):'orange',
        ('BLACKGEM', 'i'):'firebrick',
        ('BLACKGEM', 'z'):'sienna',
        ('BLACKGEM', 'q'):'black'}
    
    plt.ioff()
    #data_files_names_only = file_name.split(".csv")[0]
    # Read file
    file = Table.read(file_name, format='ascii.csv')
    if len(file)>4:
        ins = file['inst'][0]
        source_name = file['name'][0]
        out_path = f'{ins}_plots'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            
        if ins=='NEOWISE':
            ms=6
            cs=4
            ew=4
        else:
            ms=3
            cs=2
            ew=2
            
        # Create figure and axes
        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_size_inches(9,5.5)
        ax.grid(alpha=0.5)
            
        # Plot the light curve for each filter
        available_filters=list(set(file['filter']))
        available_filters=sorted(available_filters)
        for band in available_filters:
            if y_ax=='mag':
                # Remove outliers
                sigma = remove_outliers(file['mag'][file['filter']==band], method=outliers, n_std=n_std)
                                    
                # Select time, magnitudes and magnitude errors
                t_observed = np.array(file["mjd"][file['filter']==band])[sigma]
                y_observed = np.array(file['mag'][file['filter']==band])[sigma]
                uncert = np.array(file["magerr"][file['filter']==band])[sigma]
                    
                # Bin the data
                if binsize > 0:
                    t_observed, y_observed, t_err, uncert =  phot_utils.bin_lightcurve(t_observed, y_observed, yerr=uncert, 
                                                                                       binsize=binsize, mode="average")
                else:
                    t_err=None
                        
                # Select only a set limited range of points to plot
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    if binsize > 0:
                        t_err = t_err[mask]
                    
                # Plot light curve
                plot_color = color_dict.get((ins,band), 'black')
                ax.errorbar(t_observed - ref_mjd, y_observed, xerr=t_err, yerr=uncert, color=plot_color, label=band, 
                            fmt = "o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Magnitude", family = "serif", fontsize = 16)
                    
            if y_ax=='flux':
                # Remove outliers
                sigma = remove_outliers(file['flux'][file['filter']==band], method=outliers)
                    
                # Select time, magnitudes and magnitude errors
                t_observed = np.array(file["mjd"][file['filter']==band])[sigma]
                y_observed = np.array(file['flux'][file['filter']==band])[sigma]
                uncert = np.array(file["fluxerr"][file['filter']==band])[sigma]
                
                # Select only a set limited range of points to plot
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    
                # Plot light curve
                plot_color = color_dict.get((ins,band), 'black')
                ax.errorbar(t_observed - ref_mjd, y_observed, yerr=uncert, color=plot_color, label=band, 
                            fmt = "o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Flux", family = "serif", fontsize = 16)
                    
                    
        ax.set_title(f"Gaia DR3 {source_name}", weight = "bold")
        ax.set_xlabel(f"MJD - {ref_mjd} [days]", family = "serif", fontsize = 16)
        ax.tick_params(which='major', width=2, direction='out')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, direction='out')
        ax.tick_params(labelsize = 16)
        ax.minorticks_on()
        ax.invert_yaxis()
        xmin, xmax = ax.get_xlim()    
        
        ref_mjd = 58190.45
        def xconv(x):
            tyear = Time(x+ref_mjd, format="mjd")
            return tyear.jyear # days to years
        xmin2 = xconv(xmin)
        xmax2 = xconv(xmax)
        
        ay2 = plt.twiny()
        ay2.minorticks_on()
        ay2.tick_params(which='major', width=2, direction='out')
        ay2.tick_params(which='major', length=7)
        ay2.tick_params(which='minor', length=4, direction='out')
        ay2.tick_params(labelsize = 16)
        ay2.set_xlim([xmin2, xmax2])
        ay2.set_xlabel('Year', fontsize=16)
        ay2.ticklabel_format(useOffset=False)
            
                
        colors_for_filters = [color_dict.get((ins, filt), None) for filt in available_filters]
                
        ax.legend(loc = "upper right", ncols = len(available_filters), labelcolor = colors_for_filters, 
                  shadow = True, columnspacing = 0.8, handletextpad = 0.5, handlelength = 1, markerscale = 1.5, 
                  prop = {"weight" : "bold"})
        
        if savefig == True:
            plt.savefig(f'{out_path}/{source_name}.png', bbox_inches = "tight", format = "png")
        
        if plot == True:
            plt.show()
        
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


def lc_folding(name, time, y, uncert, best_freq, ax, cycles=1):
    '''
    Folds given light curve at a certain frequency.
    
    Parameters
    ----------
    name: string
        Name of the source (e.g. 'Gaia DR3 123')
    time: array of floats
        Array with the time values.
    y: array of floats
        Array with the magnitude values.
    uncert: array of floats
        Array with the magnitude errors.
    best_freq: float
        Value of the frequency to fold the light curve at.
    ax: matplotlib axes
        Where to plot the folded light curve
    cycle: float, optional
        How many phases are shown.
    Returns
    -------
    None

    '''
    
    # Remove outliers
    sigma = remove_outliers(y)
    
    # Variables to input into the FINKER script
    time = np.array(time)[sigma]
    y = np.array(y)[sigma]
    if uncert is not None:
        uncert = np.array(uncert)[sigma]
    
    # Set initial time
    min_index = np.argmax(y)
    time = time - time[min_index]
    
    
    # Plot folded light curve
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.errorbar((time * best_freq) % cycles, y,yerr=uncert, fmt='o', ecolor='black', capsize=2, elinewidth=2,
                 markersize=2, markeredgewidth=2, alpha=0.8, color='black',zorder=0)
    ax.set_xlabel('Phase', fontsize=18)
    ax.set_ylabel(r'Magnitude', fontsize=18)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=14)
    if ((24*60)/best_freq)<60:
        ax.set_title(f'{name}\nFreq: {best_freq}'+ '$d^{-1}$'+f'; period: {(24*60) / best_freq:.3f}' + r'$ \, min$',
                     fontsize=18)
    elif ((((24*60)/best_freq)>60) and (((24*60)/best_freq)<(24*60))):
        ax.set_title(f'{name}\nFreq: {best_freq}'+ '$d^{-1}$'+f'; period: {24 / best_freq:.3f}' + r'$ \, hr$',
                     fontsize=18)
    else:
        ax.set_title(f'{name}\nFreq: {best_freq}'+ '$d^{-1}$'+f'; period: {1/best_freq:.3f}' + r'$ \, d$',
                     fontsize=18)
   
        
    

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
    sigma = remove_outliers(y)
    
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