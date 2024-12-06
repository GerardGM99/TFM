# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:49:06 2024

@author: Gerard Garcia
"""

from gaiaxpy import calibrate
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np
import os
from PyAstronomy import pyasl
from astropy.modeling import models
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
from scipy.constants import h, c, k
from math import pi
from astroquery.gaia import Gaia
from scipy.special import wofz
from coord_utils import sky_xmatch
from extinction import fitzpatrick99, remove, calzetti00
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_lines
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.analysis import equivalent_width
from scipy.integrate import quad

def bin_spectrum(wavelengths, fluxes, bin_size):
    """Bin the spectrum by averaging over specified bin_size."""
    # Convert to NumPy arrays if inputs are pandas Series
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    
    # Calculate the number of bins
    num_bins = len(wavelengths) // bin_size
    
    # Reshape and average within bins
    binned_wavelengths = np.mean(wavelengths[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
    binned_fluxes = np.mean(fluxes[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
    
    return binned_wavelengths, binned_fluxes


def plotter(x, y, figsize=(10,8), plttype='plot', ax=None, xlabel='', ylabel='', title='',
            xmin=None, xmax=None, ylim=None, xinvert=False, yinvert=False, legend=False, 
            show=True, savepath=None, saveformat='png', **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if xmin is not None:
        xminmask = (xmin<=x)
        x = x[xminmask]
        y = y[xminmask]
        # ax.set_xlim(left=xlim[0], right=xlim[1])
    if xmax is not None:
        xmaxmask = (x<=xmax)
        x = x[xmaxmask]
        y = y[xmaxmask]
    if ylim is not None:
        ymask = (ylim[0]<=y) & (y<=ylim[1])
        y = y[ymask]
        x = x[ymask]
        # ax.set_ylim(bottom=ylim[0], top=ylim[1])
           
    if plttype == 'plot':
        ax.plot(x, y, **kwargs)
    elif plttype == 'scatter':
        ax.scatter(x, y, **kwargs)
    
    ax.set_xlabel(xlabel, fontsize=16, fontfamily='serif')
    ax.set_ylabel(ylabel, fontsize=16,  fontfamily='serif')
    ax.set_title(title, fontsize=16,  fontfamily='serif')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')

    
    if yinvert:
        ax.invert_yaxis()
    if xinvert:
        ax.invert_xaxis()
    
    if legend:
        ax.legend(fontsize=14)
    
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", format=saveformat)
        
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

def spec_file_convert(input_filename):
    
    data = fits.getdata(input_filename)
    header = fits.getheader(input_filename)
    
    naxis1 = len(data)
    crpix1 = header['crpix1'] * u.pixel
    cunit1 = 1 * u.Unit(header['cunit1'])
    crval1 = header['crval1'] * u.Unit(cunit1)
    cdelt1 = header['cdelt1'] * u.Unit(cunit1) / u.pixel
    wavelength = crval1 + ((np.arange(naxis1) + 1)*u.pixel - crpix1) * cdelt1
    
    df = pd.DataFrame({'wavelength':wavelength, 'flux':data})
    
    out = input_filename.split('.')[0]
    df.to_csv(f'{out}.txt', sep=' ', index=False)


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
        
def Gaia_XP(id_list, out_path=None, plot=False, ax=None):
    '''
    Plot a Gaia XP spectrum

    Parameters
    ----------
    id_list : list of strings
        List of Gaia DR3 IDs.
    out_path : string, optional
        If a path is given, the plot is saved there. The default is None.
    plot : bool, optional
        If True show the plot. The default is False.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The Axes object to plot on. If None, a new figure is created. The default is None.

    Returns
    -------
    None.

    '''
    # Balmer lines (nm)
    H_alfa = 656.3
    H_beta = 486.1
    H_gamma = 434.1
    
    calibrated_spectra, sampling = calibrate(id_list, save_file=False)

    for i in range(len(calibrated_spectra)):
        source = calibrated_spectra.iloc[[i]]
        ide = source['source_id'].iloc[0]
        
        if ax is None:
            fig, ax = plt.figure(figsize=(14, 6))
        
        ax.errorbar(sampling, np.array(source['flux'])[0], yerr=np.array(source['flux_error'])[0], 
                     fmt=".-", color="k", label="DR3")
        ax.axvline(x=H_alfa, color='r')
        ax.axvline(x=H_beta, color='c')
        ax.axvline(x=H_gamma, color='darkorchid')
        # labels, text...
        ax.set_xlabel("Rest Wavelength [nm]", fontsize=18)
        ax.set_ylabel("Flux [W nm$^{-1}$ m$^{-2}$]", fontsize=18)
        ax.set_title(ide, fontsize=18)
        
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(660, 0.95, "H$\\alpha$", transform=trans, fontdict={'fontsize': 14})
        ax.text(490, 0.95, "H$\\beta$", transform=trans, fontdict={'fontsize': 14})
        ax.text(438, 0.95, "H$\\gamma$", transform=trans, fontdict={'fontsize': 14})
        
        if plot:
            plt.show()
        
        if out_path is not None:
            plt.savefig(f'{out_path}/{ide}.png', bbox_inches="tight", format="png")
        
        if ax is None:
            plt.close()
        

def Gaia_rvs(id_list, rv_table=None, plot=True, out_dir=None):
    '''
    RVS from Gaia using datalink service.

    Parameters
    ----------
    id_list : list
        List with the Gaia DR3 source_ids.
    rv_table : astropy.table, optional
        Table with the Gaia radial velocity for each source. If given, the rv
        and expected SNR are shown in the plot. The default is None.
    plot : bool, optional
        If True, the plot is shown. The default is True.
    out_dir : string, optional
        Path to the directory where to store the RVS plots. The default is 
        None.

    Returns
    -------
    out_df : pandas.DataFrame
        Dataframe containing the x and y axis (wavelengths and fluxes) for each
        source's RVS.

    '''
    
    # Ca triplet
    CaT = [849.8, 854.2, 866.2]
    
    # Load data (https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products#datalink_jntb_get_all_prods)
    retrieval_type = 'RVS'         
    data_structure = 'COMBINED'   
    data_release   = 'Gaia DR3'
    
    datalink = Gaia.load_data(ids=id_list, data_release = data_release, retrieval_type=retrieval_type, 
                              data_structure = data_structure, verbose = False, output_file = None)
    dl_key   = 'RVS_COMBINED.xml'
    source_ids  = [product.get_field_by_id("source_id").value for product in datalink[dl_key]]
    tables      = [product.to_table()                         for product in datalink[dl_key]]
    
    x_axis=[]
    y_axis=[]
    # Plot RV Spectra for each source in id_list
    for inp_table, ids in zip(tables,source_ids):
        wavelenght = inp_table['wavelength']
        flux = inp_table['flux']
        x_axis.append(wavelenght)
        y_axis.append(flux)
        if plot:
            fig, ax = plt.subplots(figsize=(12,8))
            ax.plot(wavelenght, flux, color='k')
            for wl in CaT:
                plt.axvline(wl, linestyle='--', alpha=0.7, color='blue')
            ax.set_xlabel(f'Rest Wavelength [{inp_table["wavelength"].unit}]', fontsize=18)
            ax.set_ylabel('Normalized Flux', fontsize=18)
            ax.tick_params(axis='both', labelsize=18)
            if rv_table is not None:
                rv = rv_table['radial_velocity'][rv_table['SOURCE_ID']==ids][0]
                SNR = rv_table['rv_expected_sig_to_noise'][rv_table['SOURCE_ID']==ids][0]
                ax.set_title(f'RV: {rv:.3f} km/s; Expected SNR: {SNR:.2f}', fontsize=20)
                plt.suptitle(f'Gaia DR3 {ids}', weight='bold', fontsize=20)
            else:
                ax.set_title(f'Gaia DR3 {ids}', weight='bold', fontsize=20)
                
            if out_dir is not None:
                plt.savefig(f'{out_dir}/{ids}_RVS.png', bbox_inches = "tight", format = "png")
            plt.show()
            plt.close()

    # Output a dataframe with the x and y axis of each source's RVS
    out_df = pd.DataFrame({'IDs':source_ids, 'wavelenghts':x_axis, 'fluxes':y_axis})
    return out_df

def fluxcalib_with_Gaia(file_to_calibrate, gaia_id, save_path, plot=False):
    # Observations    
    new_spec = pd.read_csv(file_to_calibrate, sep=' ')
    max_wl = max(new_spec['wavelength'])
    min_wl = 3700
    raw_wl = new_spec['wavelength']
    raw_flux = new_spec['flux']
    mask = (raw_wl<max_wl)&(raw_wl>min_wl)
    raw_wl = raw_wl[mask]
    raw_flux = raw_flux[mask]
    
    # Gaia
    calibrated_spectra, sampling = calibrate([gaia_id], save_file=False)
    # Change units to A, and to to 1e-17 erg/s/cm^2/A
    std_wl = sampling * 10
    std_flux = calibrated_spectra['flux'].iloc[0] * 100 * 1e17
    mask = (std_wl>min_wl)&(std_wl<max_wl)
    std_wl = std_wl[mask]
    std_flux = std_flux[mask]
    
    # Smoothe both spectrums
    # sigma = 50 #Smoothing. Larger values may not be good to get the initial and final fluxes.
    # ams_per_pix = np.abs(np.median(raw_wl[1:]-raw_wl[0:-1]))
    # sigma = sigma / ams_per_pix
    smoothed_gaia_flux = gaussian_filter(std_flux, sigma=20)
    smoothed_obs_flux = gaussian_filter(raw_flux, sigma=50)

    #Interpolate the raw_std in the same wavelength bins as the absolute flux spectrum. 
    # abs_flux_std = np.interp(std_wl, new_spec['wavelength'], new_spec['flux'])
    interp_gaia_flux = interp1d(std_wl, smoothed_gaia_flux, bounds_error=False, fill_value="extrapolate")
    interpolated_gaia_flux = interp_gaia_flux(raw_wl)

    factor = interpolated_gaia_flux/smoothed_obs_flux
    
    df = pd.DataFrame({'wavelength':raw_wl, 'flux':raw_flux*factor})
    df.to_csv(save_path, sep=' ', index=False)
    
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(raw_wl, interpolated_gaia_flux)
        ax.plot(raw_wl, raw_flux)
        ax.plot(raw_wl, smoothed_obs_flux)
        ax.plot(raw_wl, raw_flux*factor)

        # ax.set_ylim(bottom=-10, top=1200)

        # labels, text...
        ax.set_xlabel("Rest Wavelength [Å]", fontsize=16, fontfamily='serif')
        ax.set_ylabel("Flux [$10^{-17}$ erg s$^{-1}$ A$^{-1}$ cm$^{-2}$]", fontsize=16,  fontfamily='serif')
        ax.set_title(gaia_id, fontsize=16,  fontfamily='serif')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')

        plt.show()
        plt.close()
        
        #----#
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(raw_wl, interpolated_gaia_flux/np.median(interpolated_gaia_flux))
        ax.plot(raw_wl, raw_flux/np.median(raw_flux))
        ax.plot(raw_wl, smoothed_obs_flux/np.median(smoothed_obs_flux))
        ax.plot(raw_wl, raw_flux*factor/np.median(raw_flux*factor))

        # ax.set_ylim(bottom=-10, top=1200)

        # labels, text...
        ax.set_xlabel("Rest Wavelength [Å]", fontsize=16, fontfamily='serif')
        ax.set_ylabel("Normalized Flux", fontsize=16,  fontfamily='serif')
        ax.set_title(gaia_id, fontsize=16,  fontfamily='serif')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')

        plt.show()
        plt.close()
        
        #----#
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(raw_wl, raw_flux*factor, c='k')
        ax.plot(std_wl, std_flux)
        ax.plot(raw_wl, interpolated_gaia_flux)
        ax.plot(raw_wl, gaussian_filter(raw_flux*factor, sigma=50))
        
        
        ax.set_xlabel("Rest Wavelength [Å]", fontsize=16, fontfamily='serif')
        ax.set_ylabel("Flux [$10^{-17}$ erg s$^{-1}$ A$^{-1}$ cm$^{-2}$]", fontsize=16,  fontfamily='serif')
        ax.set_title(gaia_id, fontsize=16,  fontfamily='serif')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', labelsize=14, direction='in')

        # labels, text...
        ax.set_xlabel("Rest Wavelength [Å]", fontsize=18)
        ax.set_ylabel("Flux [$10^{-17}$ erg s$^{-1}$ A$^{-1}$ cm$^{-2}$]", fontsize=18)

        plt.show()
        plt.close()

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def show_lines(ax, lines_file, xrange, priority):
    '''
    Draws spectral lines in the given axes.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes where to draw the lines.
    lines_file : string
        Path to a text file with the rest wavelenght, the name and the priority
        of the spectral lines.
    xrange : tuple of floats
        Tuple with the left and right limits of the wavelength axis.
    priority : list of ints
        The drawn lines are the ones with the given priorities.

    Returns
    -------
    None.

    '''
    # cmap = plt.get_cmap('rainbow')
    if lines_file is not None:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        lines = Table.read(lines_file, format='ascii')
        lines_sorted = lines[np.argsort(lines['wavelength'])]
        prev_line_wl = None
        prev_line_wl2 = None
        prev_ymax = 0.83+0.08
        # prev_ymax2 = 0.75
        delta_perce2 = 1
        for line, wl, prio, color in zip(lines_sorted['line'], lines_sorted['wavelength'], lines_sorted['prio'], lines_sorted['color']):
            if (xrange[0] <= wl <= xrange[1]) & (prio in priority):
                # color = cmap(float(np.where(unique_priorities == prio)[0]) / (num_priorities - 1))
                if prev_line_wl is not None:
                    delta_wl = wl - prev_line_wl
                    delta_perce = delta_wl / (xrange[1] - xrange[0])
                    if prev_line_wl2 is not None:
                        delta_wl2 = wl - prev_line_wl2
                        delta_perce2 = delta_wl2 / (xrange[1] - xrange[0])
                    if (delta_perce < 0.05) & (prev_ymax == 0.83+0.08) & (delta_perce2 > 0.02):
                        ymax = 0.76+0.08
                    elif (delta_perce < 0.05) & (prev_ymax == 0.83+0.08) & (delta_perce2 < 0.02):
                        ymax = 0.72+0.08
                    else:
                        ymax = 0.83+0.08
                else:
                    ymax=0.83+0.08
                ax.axvline(wl, ymax=ymax, ls='--', alpha=0.5, color=color, lw=1)
                ax.text(wl, ymax+0.01, line, transform = trans, fontdict={'fontsize':11}, rotation = 90, ha='center')
                prev_line_wl2 = prev_line_wl
                prev_line_wl = wl
                # prev_ymax2 = prev_ymax
                prev_ymax = ymax


def spec_plot(input_filename, norm=True, Av=None, ax=None, xmin=3900, xmax=9100, ylim=None, lines_file=None, plot=True):
    
    spec = pd.read_csv(input_filename, sep=' ')
    wavelength = spec['wavelength']
    flux = spec['flux']
    if norm is True:
        flux = flux/np.mean(flux)
    elif (norm=='region')&(xmin is not None)&(xmax is not None):
        length = xmax-xmin
        mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.4))
        flux_blue = flux[mask_blue]
        mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.4))
        flux_red = flux[mask_red]
        flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
        flux = flux/flux_mean
    name=os.path.basename(input_filename).split('.')[0].split('_')[0]
    if Av is not None:
        flux = remove(fitzpatrick99(np.array(wavelength), Av, 3.1), np.array(flux))
    plotter(wavelength, flux, figsize=(14,6), plttype='plot', ax=ax, 
            xlabel=r'Wavelength [Å]', ylabel='Normalized Flux', title=name,
            xmin=xmin, xmax=xmax, ylim=ylim, xinvert=False, yinvert=False, legend=False, 
            show=False, savepath=None, saveformat='png', color='k')
    
    if xmin is None:
        xmin = np.min(wavelength)
    if xmax is None:
        xmax = np.max(wavelength)
        
    if ax is None:
        ax=plt.gca()
    # xrange = [min(wavelength), max(wavelength)]
    if lines_file is not None:
        show_lines(ax, lines_file, xrange=[xmin, xmax], priority=['1','2'])
    
    if (xmin<6950) & (xmax>6870):
        ax.axvspan(6870, 6950, alpha=0.3, color='lightsteelblue')
    if (xmin<7700) & (xmax>7590):
        ax.axvspan(7590, 7700, alpha=0.3, color='lightsteelblue')
    if (xmin<7350) & (xmax>7150):
        ax.axvspan(7150, 7350, alpha=0.3, color='lightsteelblue')
    
    if plot:
        plt.tight_layout()
        plt.show()
        plt.close()
    
    if (norm=='region')&(xmin is not None)&(xmax is not None):
        return flux_mean

def cafos_spectra(input_filename, asciicords, xrange=[3500, 9501], calibration='flux', dered=None, 
                  lines_file=None, priority=['1'], plot=True, outdir=None, ax=None):
    '''
    Plots the CAFOS spectrum in the given file.

    Parameters
    ----------
    input_filename : string
        File with the wavelenghts and fluxes of the spectrum.
    asciicords : string
        Path to the file containing the name and coordinates of the source, and
        the Av extinction coeficient.
    xrange : tuple of floats, optional
        Tuple with the left and right limits of the wavelength axis. The default 
        is [3500, 9501].
    calibration : string or None, optional
        If flux, plots the flux calibrated spectrum, if None plots the observed
        spectrum. The default is 'flux'.
    dered : string or None, optional
        Can be set to the function used to deredden the spectrum using the 
        exctinction module. Can be 'fitz' or 'calz'. The default is None.
    lines_file : string or None, optional
        Path to a text file with the rest wavelenght, the name and the priority
        of the spectral lines. The default is None.
    priority : list of ints, optional
        The drawn lines are the ones with the given priorities. The default is 
        [1].
    plot : Boolean, optional
        If set to True, draws the plot. The default is True.
    outdir : string or None, optional
        Path where to save the spectrum as a jpg. The default is None.

    Returns
    -------
    None.

    '''
    if asciicords is not None:
        fits_file = input_filename.split(".")[0]+'.fits'
        hdu = fits.open(fits_file)
        ra = hdu[0].header['RA']
        dec = hdu[0].header['DEC']
        hdu.close()
        
        # Xmatch with the asciicoords file
        table = Table({'ra':[ra], 'dec':[dec]})
        table_coord = Table.read(asciicords, format='ascii.csv')
        column_names = ['ra', 'dec', 'RA', 'DEC', 'Gaia source ID']
        xmatch_table = sky_xmatch(table, table_coord, 1800, column_names)
        source_name = xmatch_table['Name'][0]
    else:
        source_name = os.path.basename(input_filename)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        
    if calibration == 'flux':
        data = Table.read(input_filename, format='ascii')
        
        if dered is not None:
            Av = table_coord['$A_V$'][table_coord['Gaia source ID']==int(source_name)][0]
            #'deredden' flux using Fitzpatrick (1999)
            if dered == 'fitz':
                flux = remove(fitzpatrick99(np.array(data['wavelength']), Av, 3.1), np.array(data['flux']))
            elif dered == 'calz':
                flux = remove(calzetti00(np.array(data['wavelength']), Av, 3.1), np.array(data['flux']))
            ax.set_title(f'{source_name}, CAFOS dereddened Low res spectrum, $A_v$ = {Av}', fontsize=16, weight='bold')
        else:
            flux = data['flux']
            ax.set_title(source_name+', CAFOS Low res spectrum', fontsize=16, weight='bold')
        
        ax.plot(data['wavelength'][flux>=0], flux[flux>=0], color='k')
    else:
        data = fits.getdata(input_filename)
        header = fits.getheader(input_filename)
        
        naxis2, naxis1 = data.shape
        crpix1 = header['crpix1'] * u.pixel
        cunit1 = 1 * u.Unit(header['cunit1'])
        crval1 = header['crval1'] * u.Unit(cunit1)
        cdelt1 = header['cdelt1'] * u.Unit(cunit1) / u.pixel
        wavelength = crval1 + ((np.arange(naxis1) + 1)*u.pixel - crpix1) * cdelt1

   
        ax.plot(wavelength.to(u.Angstrom), data[0], color='k')
        ax.set_title(source_name+', NO standard flux calibrated!', fontsize=16, weight='bold')
    
    if lines_file is not None:
        show_lines(ax, lines_file, xrange, priority)
    
    ax.set_xlabel('Rest Wavelength [Å]', fontsize=15)
    ax.set_ylabel(r'Flux [$ergs \: cm^{-2} \: s^{-1} \: Å^{-1}$]', fontsize=15)
    if (xrange[1] - xrange[0]) >= 2500:
        ax.set_xticks(np.arange(xrange[0], xrange[1], 500))
    ax.set_xlim(left=xrange[0], right=xrange[1])
    if xrange!=[3500, 9501]:
        xrange_mask = (xrange[0] < data['wavelength'])*(data['wavelength'] < xrange[1])
        mean = np.mean(flux[xrange_mask])
        flux_min = min(flux[xrange_mask])
        flux_max = max(flux[xrange_mask])
        ax.set_ylim(bottom=flux_min-0.15*mean, top=flux_max+0.15*mean)
    else:
        flux_max = max(flux)
        mean = np.mean(flux)
        ax.set_ylim(top=flux_max+1.2*mean)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if plot:
        plt.tight_layout()
        plt.show()
        
    if outdir is not None:
        if dered:
            plt.savefig(f'{outdir}/{source_name}_dereddened.png', bbox_inches = "tight", format = "png")
        else:
            plt.savefig(f'{outdir}/{source_name}.png', bbox_inches = "tight", format = "png")
        
    
    
def spectrum(wavelength, flux, title=None, Av=None, units=['Angstrom','counts'],
             lines_file=None, priority=['1'], xrange=[3500,9501], plot=True, outdir=None, ax=None, **kwargs):
    '''
    Plots the spectrum given the wavelengths and the fluxes.

    Parameters
    ----------
    wavelength : list of floats
        Spectrum wavelengths.
    flux : list of floats
        Spectrum fluxes.
    title : string, optional
       Title for the plot. Used also to name the file if saved. The default is 
       None.
    Av : float, optional
        Extinction coefficient for the dereddening. The default is None.
    units : tuple of strings, optional
        Units of the x and y axis. The default is ['Angstrom','counts'].
    lines_file : string, optional
        Path to a text file with the rest wavelenght, the name and the priority
        of the spectral lines. The default is None.
    priority : list of ints, optional
        The drawn lines are the ones with the given priorities. The default is 
        [1].
    xrange : tuple of floats, optional
         Tuple with the left and right limits of the wavelength axis. The 
         default is [3500,9501].
    plot : Boolean, optional
        If set to True, draws the plot. The default is True.
    outdir : string or None, optional
        Path where to save the spectrum as a jpg. The default is None.

    Returns
    -------
    None.

    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    if Av is not None:
        #'deredden' flux using Fitzpatrick (1999)
        flux = remove(fitzpatrick99(np.array(wavelength), Av, 3.1), np.array(flux))
        ax.set_title(f'{title}, dereddened spectrum, $A_v$ = {Av}', fontsize=16, weight='bold')
        
    ax.plot(wavelength, flux, **kwargs)
    if Av is None:
        ax.set_title(f'{title}', fontsize=16, weight='bold')
        
    if lines_file is not None:
        show_lines(ax, lines_file, xrange, priority)
        
    ax.set_xlabel(fr'Rest Wavelength [{units[0]}]', fontsize=15)
    ax.set_ylabel(fr'Flux [{units[1]}]', fontsize=15)
    if (xrange[1] - xrange[0]) >= 2500:
        ax.set_xticks(np.arange(xrange[0], xrange[1], 500))
    ax.set_xlim(left=xrange[0], right=xrange[1])
    if xrange!=[3500, 9501]:
        xrange_mask = (xrange[0] < wavelength)*(wavelength < xrange[1])
        flux_min = min(flux[xrange_mask])
        flux_max = max(flux[xrange_mask])
        mean = np.mean(flux[xrange_mask])
        ax.set_ylim(bottom=flux_min-0.15*mean, top=flux_max+0.15*mean)
    else:
        flux_max = max(flux)
        mean = np.mean(flux)
        ax.set_ylim(top=flux_max+1.2*mean)
    ax.tick_params(axis='both', which='major', labelsize=14)
        
    if plot:
        plt.tight_layout()
        plt.show()
            
    if outdir is not None:
        output_path = f'{outdir}'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        plt.savefig(f'{output_path}/{title}.png', bbox_inches = "tight", format = "png")
    # plt.close()
    

def lamost_spectra(input_filename, asciicords, xrange=[3500, 9250], dered=None, extra_med=False,
                   lines_file=None, priority=['1'] ,plot=True, outdir=None, ax=None, color='k'):
    '''
    Plots the LAMOST spectrum in the given file.

    Parameters
    ----------
    input_filename : string
        File with the wavelenghts and fluxes of the spectrum.
    asciicords : string
        Path to the file containing the name and coordinates of the source, and
        the Av extinction coeficient.
    xrange : tuple of floats, optional
        Tuple with the left and right limits of the wavelength axis. The default 
        is [3500, 9501].
    dered : string or None, optional
        Can be set to the function used to deredden the spectrum using the 
        exctinction module. Can be 'fitz' or 'calz'. The default is None.
    lines_file : string or None, optional
        Path to a text file with the rest wavelenght, the name and the priority
        of the spectral lines. The default is None.
    priority : list of ints, optional
        The drawn lines are the ones with the given priorities. The default is 
        [1].
    plot : Boolean, optional
        If set to True, draws the plot. The default is True.
    outdir : string or None, optional
        Path where to save the spectrum as a jpg. The default is None.

    Returns
    -------
    None.
    '''

    hdu = fits.open(input_filename)
    ra = hdu[0].header['RA']
    dec = hdu[0].header['DEC']
    date = hdu[0].header['MJD']
    if os.path.basename(input_filename).startswith('med'):
        resolu = 'Med. res.'
    else:
        resolu = 'Low res.'
       
    if asciicords is not None:
        # Xmatch with the asciicoords file
        table = Table({'ra':[ra], 'dec':[dec]})
        table_coord = Table.read(asciicords, format='ascii.csv')
        column_names = ['ra', 'dec', 'RA', 'DEC', 'Gaia source ID']
        xmatch_table = sky_xmatch(table, table_coord, 1800, column_names)
        source_name = xmatch_table['Name'][0]
    else:
        source_name = hdu[0].header['OBJNAME']
    
    if resolu ==  'Low res.':
        data = hdu[1].data
        hdr = hdu[1].header
        dirty_mask = (data['WAVELENGTH'][0] < 9000)*(data['FLUX'][0] > 375)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        
        if dered:
            #'deredden' flux using Fitzpatrick (1999)
            Av = table_coord['$A_V$'][table_coord['Gaia source ID']==int(source_name)][0]
            flux = remove(fitzpatrick99(np.array(data['WAVELENGTH'][0][dirty_mask]).astype('double'), Av, 3.1), 
                          np.array(data['FLUX'][0][dirty_mask]).astype('double'))
            ax.set_title(f'{source_name}, LAMOST dereddened Low res spectrum, $A_v$ = {Av}', fontsize=16, weight='bold')
        else:
            flux = data['FLUX'][0][dirty_mask]
        
        ax.plot(data['WAVELENGTH'][0][dirty_mask], flux, color=color)
        ax.set_title(f'{source_name}, LAMOST {resolu} spectrum', fontsize=16, weight='bold')
            
        if lines_file is not None:
            show_lines(ax, lines_file, xrange, priority)
            
        ax.set_xlabel(r'Rest Wavelength [Å]', fontsize=15)
        ax.set_ylabel('Flux [number of counts]', fontsize=15)
        if (xrange[1] - xrange[0]) >= 2500:
            ax.set_xticks(np.arange(xrange[0], xrange[1], 500))
        ax.set_xlim(left=xrange[0], right=xrange[1])
        if xrange!=[3500, 9250]:
            xrange_mask = (xrange[0] < data['WAVELENGTH'][0][dirty_mask])*(data['WAVELENGTH'][0][dirty_mask] < xrange[1])
            flux_min = min(flux[xrange_mask])
            flux_max = max(flux[xrange_mask])
            ax.set_ylim(bottom=flux_min-50, top=flux_max+50)
        else:
            flux_max = max(flux)
            ax.set_ylim(top=flux_max+200)
        ax.tick_params(axis='both', which='major', labelsize=14)
            
        if plot:
            plt.tight_layout()
            plt.show()
                
        if outdir is not None:
            output_path = f'{outdir}/LAMOST-L_spectra'
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            plt.savefig(f'{output_path}/{source_name}_{resolu}_{date}MJD.png', bbox_inches = "tight", format = "png")
                
    
    elif resolu == 'Med. res.':
        single_exposures_B = []
        se_names_B = []
        single_exposures_R = []
        se_names_R = []
        for i in range(1, len(hdu)):
            data = hdu[i].data
            hdr = hdu[i].header
            band = hdr['EXTNAME']
            
            if band.startswith('COADD'):
                if ax is None:
                    fig, ax = plt.subplots(figsize=(12, 5))
                   
                ax.plot(data['WAVELENGTH'][0], data['FLUX'][0], color='k')
                ax.set_title(f'{source_name}-{band}, LAMOST {resolu} spectrum', fontsize=16, weight='bold')
                    
                ax.set_xlabel('Rest Wavelength [Å]', fontsize=15)
                ax.set_ylabel('Flux [number of counts]', fontsize=15)
                if band == 'COADD_B':
                    continue
                    xrange=[4850, 5401]
                    ax.set_xticks(np.arange(4900, 5401, 50))
                    ax.set_xlim(left=xrange[0], right=xrange[1])
                elif band == 'COADD_R':
                    xrange=[6200, 6901]
                    ax.set_xticks(np.arange(6200, 6900, 50))
                    ax.set_xlim(left=xrange[0], right=xrange[1])
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if lines_file is not None:
                    show_lines(ax, lines_file, xrange, priority)
                    
                if plot:
                    plt.tight_layout()
                    plt.show()
                        
                if outdir is not None:
                    output_path = f'{outdir}/LAMOST-M_spectra'
                    if not os.path.isdir(output_path):
                        os.makedirs(output_path)
                    plt.savefig(f'{output_path}/{source_name}_{resolu}_{band}_{date}MJD.png', bbox_inches = "tight", format = "png")
            
            elif band.startswith('B'):
                single_exposures_B.append(data)
                se_names_B.append(band)
            elif band.startswith('R'):
                single_exposures_R.append(data)
                se_names_R.append(band)
        
        # Single exposures band B
        if extra_med:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            
            for spec, spec_name in zip(single_exposures_B, se_names_B):
                ax2.plot(spec['WAVELENGTH'][0], spec['FLUX'][0], label=spec_name)
            
            plt.title(f'{source_name}, Single exposures LAMOST {resolu} spectra', fontsize=16, weight='bold')
            ax2.set_xlabel('Rest Wavelength [Å]', fontsize=15)
            ax2.set_ylabel('Flux [number of counts]', fontsize=15)
            ax2.set_xticks(np.arange(4900, 5401, 50))
            ax2.set_xlim(left=4850, right=5401)
            ax2.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(loc='upper left', fontsize=15)
            
            if plot:
                plt.tight_layout()
                plt.show()
                    
            if outdir is not None:
                output_path = f'{outdir}/LAMOST-M_spectra'
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                plt.savefig(f'{output_path}/{source_name}_B_{date}MJD.png', bbox_inches = "tight", format = "png")
            
            # Single exposures band R
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            
            for spec, spec_name in zip(single_exposures_R, se_names_R):
                ax3.plot(spec['WAVELENGTH'][0], spec['FLUX'][0], label=spec_name)
            
            plt.title(f'{source_name}, Single exposures LAMOST {resolu} spectra', fontsize=16, weight='bold')
            ax3.set_xlabel('Rest Wavelength [Å]', fontsize=15)
            ax3.set_ylabel('Flux [number of counts]', fontsize=15)
            ax3.set_xticks(np.arange(6200, 6900, 50))
            ax3.set_xlim(left=6200, right=6901)
            ax3.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(loc='upper left', fontsize=15)
            
            if plot:
                plt.tight_layout()
                plt.show()
                    
            if outdir is not None:
                output_path = f'{outdir}/LAMOST-M_spectra'
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                plt.savefig(f'{output_path}/{source_name}_R_{date}MJD.png', bbox_inches = "tight", format = "png")
        
    hdu.close()
    
    if resolu == 'Low res.':
        return data['WAVELENGTH'][0][dirty_mask], flux
    
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def spec_velocity(rest_wl, wavelengths, fluxs, site=None, RA=None, DEC=None, obs_time=None, 
                  ax=None, legend=True, line='line', xlim=None, ylim=None, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    
    if site is not None:
        location = EarthLocation.of_site(site)
        sc = SkyCoord(ra=RA, dec=DEC, unit='deg')
        heliocorr = sc.radial_velocity_correction('heliocentric', obstime=Time(obs_time), 
                                                  location=location)  
        heliocorr = heliocorr.to(u.km/u.s).value
        # print(heliocorr)
    else:
        heliocorr = 0
    
    c = 299792.458 # km/s
    velocity = heliocorr + c*(wavelengths-rest_wl)/wavelengths
    # mean_flux = np.mean(fluxs)
    # flux_continum = fluxs[(abs(velocity)>0.6*xlim[1])&(abs(velocity)<2*xlim[1])]
    # mean_flux = np.mean(flux_continum)
    if xlim is not None:
        xmask = (xlim[0]<=velocity) & (velocity<=xlim[1])
        velocity = velocity[xmask]
        fluxs = fluxs[xmask]
        # ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ymask = (ylim[0]<=fluxs) & (fluxs<=ylim[1])
        fluxs = fluxs[ymask]
        velocity = velocity[ymask]
        # ax.set_ylim(bottom=ylim[0], top=ylim[1])
    flux_continum = fluxs[abs(velocity)>0.6*xlim[1]]
    mean_flux = np.mean(flux_continum)
    if line=='line':
        ax.plot(velocity, fluxs/mean_flux, **kwargs)
    elif line=='bin':
        ax.step(velocity, fluxs/mean_flux, where='mid', **kwargs)
    elif line=='scatter':
        ax.scatter(velocity, fluxs/mean_flux, **kwargs)
    ax.set_xlabel('Velocity [km/s]', fontsize=16)
    ax.set_ylabel('Normilized Flux', fontsize=16)
    # ax.set_xticks(np.arange(6200, 6900, 50))
    ax.tick_params(axis='both', labelsize=16)
    
    if legend:
        ax.legend(fontsize=14, frameon=False, fancybox=False)
        

def classification_grid(wavelengths, fluxes, Obj_name, bin_size=None, velocity_range=1, #xrange=[5, 5], 
                        site=None, RA=None, DEC=None, obs_time=None,
                        savepath=None):
       
    classification_lines = {'H':[4861, 6563], 
                            'He I':[4388, 4922, 5016],
                            'He I(weak)':[4144, 4471, 5876, 6678, 7065],
                            'He II':[4200, 4542, 4686],
                            'Ca II (K&H)':[3933.663, 3968.468], 
                            'Ca II (triplet)':[8498.03, 8542.09, 8662.14],
                            'Na I':[5890, 5896], #5860
                            'Mg II':[4481.15], 
                            'Fe I':[3860, 5167], #8688
                            'Fe II':[4233, 4924, 5016],
                            'Si III':[4552, 4568, 4575], #OB, early to mid B
                            '[O I]':[5577.34, 6300.304, 6363.776]} #Herbig Ae/Be, B[e], Novae
    
    
    fig = plt.figure(constrained_layout=True, figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.20, hspace=0.20)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[1,2])
    ax6 = fig.add_subplot(gs[2,0])
    ax7 = fig.add_subplot(gs[2,1])
    ax8 = fig.add_subplot(gs[2,2])
    axes=[ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    transitions = [['H'], ['He I'], ['He II'], ['Na I'], ['Ca II (K&H)'], ['Ca II (triplet)'], ['Fe II', 'Mg II'], ['Si III'], ['[O I]']] #, 'HeI(weak)'
    
    if velocity_range==1:
        vel_range = {'H':[-1000, 1000],
                     'He I':[-700, 700],
                     'He II':[-200, 200], 
                     'Na I':[-200, 200],
                     'Ca II (K&H)':[-200, 200], 
                     'Ca II (triplet)':[-200, 200],
                     'Fe II&Mag II':[-200, 200],
                     'Si III':[-200, 200],
                     '[O I]':[-200, 200]}
    else:
        vel_range = {'H':[-1000, 1000],
                     'He I':[-1000, 1000],
                     'He II':[-1000, 1000], 
                     'Na I':[-1000, 1000],
                     'Ca II (K&H)':[-1000, 1000], 
                     'Ca II (triplet)':[-1000, 1000],
                     'Fe II&Mag II':[-600, 600],
                     'Si III':[-500, 500],
                     '[O I]':[-500, 500]}
    
    color_palette = ['#004E64', '#00A5CF', '#9600FF', '#70E000', '#7AE582']
    # color_palette = ['#00B8FF', '#4900FF', '#9600FF', '#FF00C1']
    
    for ax, transition, key_xrange in zip(axes, transitions, vel_range.keys()):
        for sub in transition:
            rest_wls = classification_lines.get(sub)
            for idx, rest_wl in enumerate(rest_wls):
                # mask = (wavelengths>rest_wl-xrange[0]) & (wavelengths<rest_wl+xrange[1])
                label=fr'{rest_wl} Å'
                if transition == ['Fe II', 'Mg II']:
                    sub = 'Fe II and Mg II'
                    if rest_wl == 4481.15:
                        label = fr'MgII {rest_wl} Å'
                        color = '#70E000'
                    else:
                        label = fr'FeII {rest_wl} Å'
                        color = color_palette[idx % len(color_palette)]
                else:
                    color = color_palette[idx % len(color_palette)]
                xrange = vel_range.get(key_xrange)
                if bin_size is not None:
                    # bin_size = 10
                    binned_wavelengths, binned_fluxes = bin_spectrum(wavelengths, fluxes, bin_size)
                    spec_velocity(rest_wl, binned_wavelengths, binned_fluxes, site=site, RA=RA, DEC=DEC, obs_time=obs_time, 
                                  ax=ax, legend=True, line='bin', label=label, xlim=xrange, color=color)
                else:
                    spec_velocity(rest_wl, wavelengths, fluxes, site=site, RA=RA, DEC=DEC, obs_time=obs_time, 
                                  ax=ax, legend=True, line='line', label=label, xlim=xrange, color=color)
                
        ax.tick_params(axis='both', direction='in')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(sub, fontsize=16, weight='bold')        
    
    fig.suptitle(Obj_name, fontsize=16, weight='bold', y=0.91)
    fig.text(0.5, 0.09, r'Velocity [km/s]', ha='center', va='center', fontsize=18)
    fig.text(0.07, 0.5, r'Normilized Flux', ha='center', 
             va='center', rotation='vertical', fontsize=18)
    
    plt.tight_layout()    
    if savepath is not None:
        plt.savefig(f'{savepath}/specgrid_{Obj_name}.png', bbox_inches = "tight", format = "png")
    
    plt.show()
    plt.close()
    
def fits2grid(input_filename, site=None, xlim=None, lines_file=None):
    
    spec_file_convert(input_filename)
    
    header = fits.getheader(input_filename)
    RA=header['RA']
    DEC=header['DEC']
    obs_time=header['DATE']
    obj_name=header['OBJECT']
    
    path=os.path.dirname(input_filename)
    file = os.path.basename(input_filename).split('.')[0] + '.txt'
    spec_plot(os.path.join(path, file), xlim=xlim, lines_file=lines_file, norm=False)
    spectrum = pd.read_csv(os.path.join(path, file), sep=' ')
    wavelengths = spectrum['wavelength']
    fluxes = spectrum['flux']
    
    plotdir = os.path.join(path,'plots')
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    classification_grid(wavelengths, fluxes, obj_name+os.path.basename(input_filename), 
                        site=site, RA=RA, DEC=DEC, obs_time=obs_time,
                        savepath=plotdir)

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def find_lines(wavelength, flux, xmin, xmax, norm=True, smooth=None, threshold=1, noise_range=None, plot=False):
   
    mask = (wavelength>xmin)&(wavelength<xmax)
    
    # Normalize the flux using 60% of the range given (30% from the redder part and 30% form the bluer part)
    if norm is True:
        length = xmax-xmin
        mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.2))
        flux_blue = flux[mask_blue]
        mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.2))
        flux_red = flux[mask_red]
        flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
        flux = flux/flux_mean -1
   
    # Create specutils Spectrum1D class
    if smooth is not None:
        smoothed_flux = gaussian_filter(flux[mask], sigma=smooth)
        spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=smoothed_flux * u.Unit('erg / (cm2 s Å)') )
    else:
        spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') )
        
    # Find lines using specutils find_lines_derivative or find_lines_threshold
    if noise_range is None:
        lines = find_lines_derivative(spectrum, flux_threshold=threshold)
    else:
        noise_region = SpectralRegion(noise_range[0]*u.angstrom, noise_range[1]*u.angstrom)
        spectrum = noise_region_uncertainty(spectrum, noise_region)
        lines = find_lines_threshold(spectrum, noise_factor=threshold)
   
    abs_lines = lines[lines['line_type'] == 'absorption']['line_center'].value
    emi_lines = lines[lines['line_type'] == 'emission']['line_center'].value
   
    if plot:
        plotter(wavelength[mask], flux[mask], figsize=(14,6), plttype='plot', ax=None,
                xlabel=r'Wavelength [Å]', ylabel='Normalized Flux', title=None,
                xmin=xmin, xmax=xmax, ylim=None, xinvert=False, yinvert=False, legend=False,
                show=False, savepath=None, saveformat='png', color='k')
        ax=plt.gca()
        if smooth is not None:
            ax.plot(wavelength[mask], smoothed_flux, ls='--')
        for line in abs_lines:
            ax.axvline(line, color='purple', alpha=0.6)
        for line in emi_lines:
            ax.axvline(line, color='g', alpha=0.6)
        
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        plt.show()
        plt.close()
   
    return abs_lines, emi_lines

def EW(filepath, xrange):
    
    # Read spectrum
    spectrum = pd.read_csv(filepath, sep=' ')
    wavelength = np.array(spectrum['wavelength'])
    flux = np.array(spectrum['flux'])
    xmin=xrange[0]
    xmax=xrange[1]
    mask = (wavelength>xmin)&(wavelength<xmax)
    
    # Normalize
    length = xmax-xmin
    mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.2))
    flux_blue = flux[mask_blue]
    mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.2))
    flux_red = flux[mask_red]
    flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
    flux = flux/flux_mean -1
    spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, 
                          flux=flux[mask] * u.Unit('erg / (cm2 s Å)') )
    
    flux_mean = spec_plot(filepath, norm='region', ax=None, ylim=None, 
                          lines_file='data/spectral_lines.txt', plot=True, xmin=xmin, xmax=xmax)
    smooth = input('Sigma to smooth the spectrum to find the lines (skip to not smooth the spectrum): ')
    if smooth == '':
        sigma = None
    else:
        sigma = float(smooth)
    threshold = float(input('Find lines threshold: '))
    input_noise = input('Noise range (left and right separated by a comma, or None): ')
    if input_noise.lower() in ['none', 'n']:
        noise_range = None
    else:
        noise_range = np.array(input_noise.split(',')).astype(int)
    
    loop = True
    while loop is True:
        abs_lines, emi_lines = find_lines(wavelength, flux, xmin, xmax, norm=False, smooth=sigma,
                                          threshold=threshold, noise_range=noise_range, plot=True)
            
        cont = input('Correct lines? (yes/no): ')
        if cont.lower() in ['yes', 'y']:
            loop = False
        else:
            smooth = input('Sigma to smooth the spectrum to find the lines (skip to not smooth the spectrum): ')
            if smooth == '':
                sigma = None
            else:
                sigma = float(smooth)
            threshold = float(input('Find lines threshold: '))
            input_noise = input('Noise range (left and right separated by a comma, or None): ')
            if input_noise.lower() in ['none', 'n']:
                noise_range = None
            else:
                noise_range = np.array(input_noise.split(',')).astype(int) 

    gaussian_init_abs = []
    for i, line in enumerate(abs_lines):
        line_width = float(input(f'Width of absorption line {i+1} at {line} Å (window to fit a gaussian): '))
        sub_region = SpectralRegion((line-line_width/2)*u.angstrom, (line+line_width/2)*u.angstrom)
        sub_spectrum = extract_region(spectrum, sub_region)
        result = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(amplitude=result.amplitude.value* u.Unit('erg / (cm2 s Å)'), 
                                    mean=line*u.angstrom, stddev=result.stddev.value*u.angstrom)
        gaussian_init_abs.append(g_init)
        
    gaussian_init_emi = []
    for i, line in enumerate(emi_lines):
        line_width = float(input(f'Width of emission line {i+1} at {line} Å (window to fit a gaussian): '))
        sub_region = SpectralRegion((line-line_width/2)*u.angstrom, (line+line_width/2)*u.angstrom)
        sub_spectrum = extract_region(spectrum, sub_region)
        result = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(amplitude=result.amplitude.value* u.Unit('erg / (cm2 s Å)'), 
                                    mean=line*u.angstrom, stddev=result.stddev.value*u.angstrom)
        gaussian_init_emi.append(g_init)
                
    
    spectrum_min_line = spectrum
    # gaus_fit_abs = []
    flux_fit_abs = []
    ew_specutils_abs = []
    ew_quad_abs = []
    for i, gaus_line in enumerate(gaussian_init_abs):
        input_text = f'Window for the fit of absorption line {i+1} (wl axis left and right separated by a comma, or "skip"): '
        input_value = input(input_text)
        if input_value == 'skip':
            abs_lines = np.delete(abs_lines, i)
            # abs_lines.remove(abs_lines[i])
            print(f'Skipping absorption line {i+1} at {abs_lines[i]}...')
            continue
        window = np.array(input_value.split(',')).astype(float)
        ew = equivalent_width(spectrum_min_line+1* u.Unit('erg / (cm2 s Å)'), 
                              regions=SpectralRegion(window[0]*u.angstrom, window[1]*u.angstrom))
        ew_specutils_abs.append(ew.value)
        g_fit = fit_lines(spectrum_min_line, gaus_line, window=(window[0]*u.angstrom,window[1]*u.angstrom))
        y_fit = g_fit(wavelength[mask]*u.angstrom)
        # gaus_fit_abs.append(g_fit)
        flux_fit_abs.append(y_fit)
        ew_int = quad(gaussian, args=(-g_fit.amplitude.value, g_fit.mean.value, g_fit.stddev.value), a=window[0], b=window[1])
        ew_quad_abs.append(ew_int[0])
        spectrum_min_line = spectrum_min_line.add(-y_fit)
        flux_mean = spec_plot(filepath, norm='region', ax=None, ylim=None, 
                              lines_file='data/spectral_lines.txt', plot=False, xmin=xmin, xmax=xmax)
        for j, fitted_line in enumerate(flux_fit_abs):
            plt.plot(wavelength[mask], fitted_line+1* u.Unit('erg / (cm2 s Å)'))
            plt.axvspan(abs_lines[j]-ew_specutils_abs[j]/2, abs_lines[j]+ew_specutils_abs[j]/2, 
                        color='blue', alpha=0.3)
        plt.show()
        plt.close()
            
    spectrum_min_line = spectrum
    # gaus_fit_emi = []
    flux_fit_emi = []
    ew_specutils_emi = []
    ew_quad_emi = []
    for i, gaus_line in enumerate(gaussian_init_emi):
        input_text = f'Window for the fit of emission line {i+1} (wl axis left and right separated by a comma, or "skip"): '
        input_value = input(input_text)
        if input_value == 'skip':
            print(f'Skipping emission line {i+1} at {emi_lines[i]}...')
            continue
        window = np.array(input_value.split(',')).astype(float)
        ew = equivalent_width(spectrum_min_line+1* u.Unit('erg / (cm2 s Å)'), 
                              regions=SpectralRegion(window[0]*u.angstrom, window[1]*u.angstrom))
        ew_specutils_emi.append(ew.value)
        g_fit = fit_lines(spectrum_min_line, gaus_line, window=(window[0]*u.angstrom,window[1]*u.angstrom))
        y_fit = g_fit(wavelength[mask]*u.angstrom)
        # gaus_fit_emi.append(g_fit)
        flux_fit_emi.append(y_fit)
        ew_int = quad(gaussian, args=(-g_fit.amplitude.value, g_fit.mean.value, g_fit.stddev.value), a=window[0], b=window[1])
        ew_quad_emi.append(ew_int[0])
        spectrum_min_line = spectrum_min_line.add(-y_fit)
        flux_mean = spec_plot(filepath, norm='region', ax=None, ylim=None, 
                              lines_file='data/spectral_lines.txt', plot=False, xmin=xmin, xmax=xmax)
        for j, fitted_line in enumerate(flux_fit_emi):
            plt.plot(wavelength[mask], fitted_line+1* u.Unit('erg / (cm2 s Å)'))
            plt.axvspan(emi_lines[j]-ew_specutils_emi[j]/2, emi_lines[j]+ew_specutils_emi[j]/2, 
                        color='blue', alpha=0.3)
        plt.show()
        plt.close()
    
    return ew_specutils_abs, ew_quad_abs, ew_specutils_emi, ew_quad_emi

def EW2(filepath, xrange, n=100):
    
    # Read spectrum
    spectrum = pd.read_csv(filepath, sep=' ')
    wavelength = np.array(spectrum['wavelength'])
    og_wl = wavelength
    flux = np.array(spectrum['flux'])
    og_flux = flux
    xmin=xrange[0]
    xmax=xrange[1]
    mask = (wavelength>xmin)&(wavelength<xmax)
    
    spec_plot(filepath, norm=False, ax=None, ylim=None, 
              lines_file='data/spectral_lines.txt', plot=True, xmin=xmin, xmax=xmax)
    
    input_range = input('Range to show (left and right separated by a comma, or none): ')
    if input_range.lower() in ['none', 'n', '']:
        plot_range = None
    else:
        plot_range = np.array(input_range.split(',')).astype(float)
    
    print('------------------------------------------------')
    print('Chose two ranges to compute the continuum from: ')
    print('------------------------------------------------')
    input_continuum1 = input('Range 1 (left and right separated by a comma): ')
    input_continuum2 = input('Range 2 (left and right separated by a comma): ')
    
    cont_range1 = np.array(input_continuum1.split(',')).astype(float)
    cont_range2 = np.array(input_continuum2.split(',')).astype(float)
    
    # Normalize
    mask_range1 = (wavelength > cont_range1[0]) & (wavelength < cont_range1[1])
    flux_range1 = flux[mask_range1]
    mask_range2 = (wavelength > cont_range2[0]) & (wavelength < cont_range2[1])
    flux_range2 = flux[mask_range2]
    flux_median = np.median(np.array([np.median(flux_range1),np.median(flux_range2)]))
    original_flux = flux
    original_flux_median = flux_median
    flux = flux/flux_median -1
    
    mask = (wavelength>plot_range[0])&(wavelength<plot_range[1])
    wavelength=wavelength[mask]
    flux=flux[mask]
    original_flux=original_flux[mask]
    spectrum = Spectrum1D(spectral_axis=wavelength * u.angstrom, 
                          flux=flux * u.Unit('erg / (cm2 s Å)') )
    
    plotter(wavelength, flux, figsize=(14,6), plttype='plot', ax=None,
            xlabel=r'Wavelength [Å]', ylabel='Normalized Flux', title=None,
            xmin=plot_range[0], xmax=plot_range[1], ylim=None, xinvert=False, yinvert=False, legend=False,
            show=False, savepath=None, saveformat='png', color='k')
    ax=plt.gca()
    ax.axhline(flux_median, ls='--')
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    plotter(wavelength, flux, figsize=(14,6), plttype='plot', ax=None,
            xlabel=r'Wavelength [Å]', ylabel='Normalized Flux', title=None,
            xmin=plot_range[0], xmax=plot_range[1], ylim=None, xinvert=False, yinvert=False, legend=False,
            show=False, savepath=None, saveformat='png', color='k')
    ax=plt.gca()
    ax.axhline(flux_median, ls='--')
    ax.grid(True, which='both', axis='both')
    
    smoothed_flux = gaussian_filter(flux, sigma=5)
    ax.plot(wavelength, smoothed_flux, label='Smoothed spec')
    spectrum_smooth = Spectrum1D(spectral_axis=wavelength * u.angstrom, 
                          flux=smoothed_flux * u.Unit('erg / (cm2 s Å)') )
    result = estimate_line_parameters(spectrum_smooth, models.Gaussian1D())
    g_init = models.Gaussian1D(amplitude=result.amplitude.value* u.Unit('erg / (cm2 s Å)'), 
                               mean=result.mean.value*u.angstrom, stddev=result.stddev.value*u.angstrom)
    g_fit = fit_lines(spectrum, g_init, window=(plot_range[0]*u.angstrom, plot_range[1]*u.angstrom))
    y_fit = g_fit(wavelength*u.angstrom)
    ax.plot(wavelength, y_fit, label= 'Gaus fit')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    gaus_or_no = input('Use Gaus fit? (yes/no): ')
    if gaus_or_no.lower() in ['y', 'yes']:
        window = [g_fit.mean.value - 3*g_fit.stddev.value, g_fit.mean.value + 3*g_fit.stddev.value]
        # window_mask = y_fit<(-1e-2* u.Unit('erg / (cm2 s Å)'))
        # ax.axvline(min(wavelength[window_mask]))
        # ax.axvline(max(wavelength[window_mask]))
    else:
        input_window = input('Window to compute equivalent width (left and right separated by a comma): ')
        window = np.array(input_window.split(',')).astype(float)
    
    plotter(wavelength, flux, figsize=(14,6), plttype='plot', ax=None,
            xlabel=r'Wavelength [Å]', ylabel='Normalized Flux', title=None,
            xmin=plot_range[0], xmax=plot_range[1], ylim=None, xinvert=False, yinvert=False, legend=False,
            show=False, savepath=None, saveformat='png', color='k')
    ax=plt.gca()
    ax.axhline(0, ls='--') #continuum
    ax.grid(True, which='both', axis='both')
    
    ax.axvline(window[0], c='r', ls='--')
    ax.axvline(window[1], c='r', ls='--')
    
    ew = equivalent_width(spectrum+1* u.Unit('erg / (cm2 s Å)'), 
                          regions=SpectralRegion(window[0]*u.angstrom, window[1]*u.angstrom))
    
    window_length = window[1]-window[0]
    if gaus_or_no.lower() in ['y', 'yes']:
        ax.axvspan(g_fit.mean.value-ew.value, g_fit.mean.value+ew.value, color='k', alpha=0.3)
    else:
        ax.axvspan((window[0]+window_length/2)-ew.value, (window[1]-window_length/2)+ew.value, color='k', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print('----------------------------------------------------------------------------')
    print(f'Iterating {n} times to find EW with error by changing window length randomly')
    print('----------------------------------------------------------------------------')
    
    ews = []
    window_length_n = []
    flux_median_n = []
    if n != 0:
        for _ in range(n):
            # Shift the window
            shift = np.random.uniform(-0.25, 0.25)
            new_window = (window[0] - shift, window[1] + shift)
            new_window_length = new_window[1] - new_window[0]
            window_length_n.append(new_window_length)
            
            # Shift the continuum to normalize
            shift = np.random.uniform(-0.5, 0.5)
            mask_range1 = (og_wl > (cont_range1[0]+shift)) & (og_wl < (cont_range1[1]+shift))
            flux_range1 = og_flux[mask_range1]
            mask_range2 = (og_wl > (cont_range2[0]+shift)) & (og_wl < (cont_range2[1]+shift))
            flux_range2 = og_flux[mask_range2]
            flux_median = np.median(np.array([np.median(flux_range1),np.median(flux_range2)]))
            flux = original_flux/flux_median -1
            flux_median_n.append(flux_median)
            
            spectrum_i = Spectrum1D(spectral_axis=wavelength * u.angstrom, 
                                    flux=flux * u.Unit('erg / (cm2 s Å)') )
            
            ew_i = equivalent_width(spectrum_i+1* u.Unit('erg / (cm2 s Å)'), 
                                    regions=SpectralRegion(new_window[0]*u.angstrom, new_window[1]*u.angstrom))
            ews.append(ew_i.value)

        mean_ew = np.mean(ews)
        std_ew = np.std(ews)
        std_length = np.std(np.array(window_length_n)/window_length)
        std_cont = np.std(np.array(flux_median_n)/original_flux_median)
    
        return ew.value, mean_ew, std_ew, [std_length, std_cont]
    
    return ew.value
    
def NaID_extinction(ew, line='D1'):
    if line == 'D1':
        D1 = 10**(2.47*ew - 1.76)
        AVD1 = D1*3.1
        return D1, AVD1
    elif line == 'D2':
        D2 = 10**(2.16*ew - 1.91)
        AVD2 = D2*3.1
        return D2, AVD2
    elif line=='combined':
        D1_D2 = 10**(1.17*ew - 1.85)
        AVD1_D2 = D1_D2*3.1
        return D1_D2, AVD1_D2
    
def DiBs_extinction(ew, line='5780'):
    if line == '5780':
        dib5780 = 1.978*ew + 0.035
        AV5780 = dib5780*3.1
        return dib5780, AV5780
    elif line == '6614':
        dib6614 = 3.846*ew + 0.072
        AV6614 = dib6614*3.1
        return dib6614, AV6614

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# def keplerian_velocity(r, M):
#     """
#     Keplerian orbital velocity at distance r from a central object of mass M.
#     """
#     G = 6.67430e-11  # Gravitational constant, [m^3 kg^(-1) s^(-2)]
#     v_orb = np.sqrt(G * M / r)
#     return v_orb

# def disk_velocity_distribution(r_range, theta_range, M, i):
#     """
#     Computes the line-of-sight velocity for points in the disk, given a range of radii and angles.
#     """
#     velocities = []
#     for r in r_range:
#         for theta in theta_range:
#             v_orb = keplerian_velocity(r, M)
#             # Projected velocity along the line of sight
#             v_los = v_orb * np.sin(i) * np.cos(theta)
#             velocities.append(v_los)
#     return velocities

#thermal broadening 
def sig_T(T, wl):
    '''
    Standard deviation of a Gaussian spectral line due to thermal Doppler
    boradening, given a temperature T and the position of the line wl.
    '''
    kB = 1.381 * 10e-23 #Boltzmann constant, [J·K^{-1}]
    c = 299792458 #Speed of light, [m/s]
    mH = 1.6735575 * 10e-27 #Hydrogen mass, [kg]
    sig_T = np.sqrt(kB * T/(mH * c**2)) * wl
    return sig_T


def line_width(sig_int, wl, R):
    '''
    Total line width of a Gaussian spectral line. Adds the standard deviation
    due to the resolution of the spectrograph (of resolving power R) to the
    intrinsic deviation sig_int, at position of the line wl.
    '''
    sig_R = wl/(2*np.sqrt(2*np.log(2))*R)
    sigma = np.sqrt(sig_int**2 + sig_R**2)
    return sigma


def gaussian_line(height, wl, sig, wl_range, n_points):
    '''
    Generates a Gaussian spectral line at position wl, with a given height 
    and standard deviation.
    '''
    g1 = models.Gaussian1D(height, wl, sig)
    x = np.linspace(wl_range[0], wl_range[1], n_points)
    y = g1(x)
    return x, y

def voigt_line(lambda1, lambdax, gamma, T, m, R):
    '''
    Generates a line with a Voigt distribution. UNITS!!!!

    Parameters
    ----------
    lambda1 : float
        Central wavelength of the line.
    lambdax : array of floats
        Wavelength as independent variable (x axis).
    gamma : float
        Constant related to the probability of the transitions for each atom.
    T : float
        Temperature of the gas.
    m : float
        Atomic mass.
    R : float
        Resolving power of the intrument.

    Returns
    -------
    V : array of floats
        Voigt distribution (y axis).

    '''    
    # Standard deviation of the Gaussian (DOPPLER BROADENING)
    thermal_b = np.sqrt(2*k*T/m)  # m/s
    FWHM = thermal_b * lambda1/c  # A
    sigma_doppler = FWHM / (2*np.sqrt(2*np.log(2)))  # A
    
    # Broadening due to instrument
    sigma_R = lambda1/(2*np.sqrt(2*np.log(2))*R)  # A
    
    # Add the 2 sigmas together
    sigma = np.sqrt(sigma_doppler**2 + sigma_R**2)
    
    # Voigt parameters
    # u_v = c*(lambda1/lambdax - 1)
    # a = lambda1*gamma/(4*pi)
    u_v = c/thermal_b*(lambda1/lambdax - 1)
    nudop=1e10*thermal_b/lambda1
    a = gamma/(4*pi*nudop)
    
    # Voigt distribution
    # V = wofz((u_v + 1j * a)/(2*sigma)).real / (np.sqrt(2 * pi) * sigma)
    V = wofz(u_v + 1j * a).real
    return V

def add_noise(s, SNR):
    '''
    Adds random (Gaussian) noise to a signal s, given an SNR.
    '''
    noise = np.random.normal(0., np.mean(s)/SNR, s.shape)
    return s + noise 

# Shifted line (due to radial velocity)
def rv_shift(wl, v):
    '''
    Gives the shift of postion of a line at wl, due to a velocity v (Doppler
    shifting).
    '''
    c = 299792.458 #Speed of light, [km/s]
    dlambda = wl * v/c
    return dlambda

def blackbody_radiation(wavelength, T):
    '''
    Blackbody radiation following Planck's law (wavelength).
    '''
    # Plank's law (h in [J s], c in [m/s], k_b in [J/K], wavelenght in meters)
    intensity = (2 * h * c**2) / (wavelength**5) * (1 / (np.exp((h * c) / (wavelength * k * T)) - 1))
    return intensity

def model_spectral_lines(wl_range, lines, line_strenght, R, T, SNR, pix_size=None, rv=[0], plot=True, cont=False):

    #I NEED TO ADD ALL THE PARAMETERS (LOGG, METALLICITY, ALPHA)    
    
    # Pixel size & sample points
    if pix_size is not None:
        n_points = round((wl_range[1]-wl_range[0])/pix_size)
    else:
        central_wl = (wl_range[0]+wl_range[1])/2.
        res_element = central_wl/R
        pix_size = res_element/2. # We assume 2 pixels per resolution element
        print('pix size: ', pix_size)
        n_points = round((wl_range[1]-wl_range[0])/pix_size)

    # Create the template (only if rv is not 0; if rv=0, the tamplate spectra
    # is the black body spectrum at temperature T)
    if rv != [0]:
        ty = np.zeros(n_points)
        for position, height in zip(lines.values(), line_strenght):
            sigma = line_width(sig_T(T,position), position, R)
            tx, y = gaussian_line(height, position, sigma, wl_range, n_points)
            ty += y
        ty = add_noise(ty, 150)
    else:
        tx, ty = np.linspace(wl_range[0], wl_range[1], n_points), 0
    
    # Create data
    spectra = []
    for vel in rv:
        dy1 = np.zeros(n_points)
        for position, height in zip(lines.values(), line_strenght):
            dlambda = rv_shift(position, vel)
            sigma = line_width(sig_T(T,position+dlambda), position+dlambda, R)
            dx, y = gaussian_line(height, position+dlambda, sigma, wl_range, n_points)
            # dx = np.linspace(wl_range[0], wl_range[1], n_points)
            # gamma, m = 1e5, 1.6735575 * 10e-27 #PROVISIONAL
            # y = voigt_line(position+dlambda, dx, gamma, T, m, R)
            dy1 += y #sum for each line
        spectra.append(dy1)
        
    dy = sum(spectra) #sum for each velocity
    
    # Adds blackbody spectrum at temperature T
    if cont:
        wavelenghts = np.linspace(wl_range[0], wl_range[1], n_points)/1e10 # in meters
        flux = blackbody_radiation(wavelenghts, T)
        dy += flux/np.mean(flux)
        ty += flux/np.mean(flux)
    else:
        dy += 1
        ty += 1
    
    dy = add_noise(dy, SNR)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] #Colors for the axvlines
    color_cycle = iter(colors)
    # Plot template and data
    if plot:
        plt.figure(figsize=(14,7))
        plt.title(f'Resolution: {R}; Temperature: {T} K, SNR: {SNR}, RVs: {rv} km/s')
        plt.plot(dx, dy, label='Data')
        for line_v, line_k in zip(lines.values(), lines.keys()):
            color = next(color_cycle)
            plt.axvline(line_v, label=line_k, linestyle='--', alpha=0.7, color=color)
        if rv != [0]:
            plt.plot(tx, ty, ls='dotted', label='Template', color='k')
        elif (rv==[0])&(cont is True):
            plt.plot(tx, ty, ls='dotted', label='Blackbody', color='k')
        plt.xlabel('Spectral Axis') 
        plt.ylabel('Flux Axis')
        plt.grid()
        plt.legend(loc='upper left')
        plt.show()

    return tx, ty, dx, dy


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def rv_crosscorr(dx, dy, tx, ty, rvmin, rvmax, step, skipedge=0, plot=True):
    '''
    Calculates the radial velocity of a shifted line cross-correlating with a
    template, using the pyasl module function pyasl.crosscorrRV().

    Parameters
    ----------
    dx : list of floats
        Data values on the X axis (wavelengths).
    dy : list of floats
        Data values on the Y axis (fluxes).
    tx : list of floats
        Template values on the X axis (wavelengths).
    ty : list of floats
        Template values on the Y axis (fluxes).
    rvmin : float
        Minimum radial velocity for which to calculate the cross-correlation 
        function [km/s].
    rvmax : float
        Maximum radial velocity for which to calculate the cross-correlation 
        function [km/s].
    step : float
        The width of the radial-velocity steps to be applied in the calculation 
        of the cross-correlation function [km/s].
    skipedge : float, optional
        If larger zero, the specified number of bins will be skipped from the 
        begin and end of the observation. This may be useful if the template 
        does not provide sufficient coverage of the observation.
    plot : bool, optional
        If True, show the cros-correlation VS RV plot. The default is True.

    Returns
    -------
    rv[maxind] : float
        Radial velocity that maximizes the cross-correlation function.
    cc[maxind]: float
        Maximum value of the cross-correlation funciton.

    '''
    
    # Carry out the cross-correlation.
    rv, cc = pyasl.crosscorrRV(dx, dy, tx, ty, rvmin, rvmax, step, skipedge=skipedge)
    
    # Find the index of maximum cross-correlation function
    maxind = np.argmax(cc)
    
    print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
    
    if plot:
        plt.figure()
        plt.scatter(rv, cc, s=5, c='b')
        plt.plot(rv[maxind], cc[maxind], 'ro')
        plt.show()

    return rv[maxind], cc[maxind]



def rv_crosscorr_err(n_boot, wl_range, n_points, lines, line_strenght, R, T, SNR, rv=[0], skipedge=20):
    '''
    Repeats the calculation of the radial veloctiy using cross-correlation
    (rv_cross_corr()) 'n_boot' times to find a mean value of the radial
    velocity that maximizes the cross-correlation function and an error given
    by the standard deviation of all the values (bootstraping). Can find two
    velocities, a positive and a negative rv.

    Parameters
    ----------
    n_boot : int
        Number of times the cross-correlation method is used to find a mean rv
        and an error.
    wl_range : TYPE
        DESCRIPTION.
    n_points : TYPE
        DESCRIPTION.
    lines : TYPE
        DESCRIPTION.
    line_strenght : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    SNR : TYPE
        DESCRIPTION.
    rv : TYPE, optional
        DESCRIPTION. The default is [0].
    skipedge : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    '''
    
    RVs = []
    if (len(rv)==1):
        rvmin, rvmax = rv[0]-10, rv[0]+10
    elif len(rv)==2:
        rvmin, rvmax = rv[0]-10, rv[1]+10
    step = (rvmax - rvmin)/200
    for n in range(n_boot):
        if n == 0:
            tx, ty, dx, dy = model_spectral_lines(wl_range, n_points, lines, line_strenght, R, T, SNR, rv=rv, plot=True)
            found_rv, _ = rv_crosscorr(dx, dy, tx, ty, rvmin, rvmax, step, skipedge=skipedge, plot=True)
        else:
            tx, ty, dx, dy = model_spectral_lines(wl_range, n_points, lines, line_strenght, R, T, SNR, rv=rv, plot=False)
            found_rv, _ = rv_crosscorr(dx, dy, tx, ty, rvmin, rvmax, step, skipedge=skipedge, plot=False)
        RVs.append(found_rv)
    
    # Differentiates positive and negative values, in case there are two 
    # spectra shifter by two different velocities, one negative the other positive
    pos_RVs = [x for x in RVs if x > 0]
    neg_RVs = [x for x in RVs if x < 0]
    pos_RV_mean = np.mean(pos_RVs)
    pos_RV_std = np.std(pos_RVs)
    neg_RV_mean = np.mean(neg_RVs)
    neg_RV_std = np.std(neg_RVs)
    
    print('---')
    print(f'Positive radial velocity: {pos_RV_mean} '+u'\u00B1'+f' {pos_RV_std} km/s')
    print(f'Negative radial velocity: {neg_RV_mean} '+u'\u00B1'+f' {neg_RV_std} km/s')