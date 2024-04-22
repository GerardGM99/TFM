# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:49:06 2024

@author: Gerard Garcia
"""

from gaiaxpy import convert, calibrate
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

def Gaia_XP(id_list, out_path=None, plot=False):
    '''
    Plota Gaia XP spectrum

    Parameters
    ----------
    id_list : list of strings
        List of Gaia DR3 IDs.
    out_path : string, optional
        If a path is given, the plot is saved there. The default is None.
    plot : bool, optional
        If True show the plot. The default is False.

    Returns
    -------
    None.

    '''
    #Balmer lines (nm)
    H_alfa = 656.3
    H_beta = 486.1
    H_gamma = 434.1
    
    calibrated_spectra, sampling = calibrate(id_list)

    for i in range(len(calibrated_spectra)):
        source = calibrated_spectra.iloc[[i]]
        ide = source['source_id'].iloc[0]
        plt.figure(figsize=(14,6))
        plt.errorbar(sampling, np.array(source['flux'])[0], yerr=np.array(source['flux_error'])[0], 
                     fmt=".-", color="k", label = "DR3")
        plt.axvline(x = H_alfa, color = 'r')
        plt.axvline(x = H_beta, color = 'c')
        plt.axvline(x = H_gamma, color = 'darkorchid')
        #labels, text...
        plt.xlabel("Wavelength [nm]", fontsize=18)
        plt.ylabel("Flux [W nm$^{-1}$ m$^{-2}$]", fontsize=18)
        plt.title(ide, fontsize=18)
        ax = plt.gca()
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.text(660, 0.95, "H$\\alpha$", transform = trans, fontdict={'fontsize':14})
        plt.text(490, 0.95, "H$\\beta$", transform = trans, fontdict={'fontsize':14})
        plt.text(438, 0.95, "H$\\gamma$", transform = trans, fontdict={'fontsize':14})
        
        if plot==True:
            plt.show()
        
        if out_path is not None:
            plt.savefig(f'{out_path}/{ide}.png', bbox_inches = "tight", format = "png")
        
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
            ax.set_xlabel(f'Wavelength [{inp_table["wavelength"].unit}]', fontsize=18)
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

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def cafos_spectra(input_filename, asciicords, calibration='flux', lines_file=None, plot=True, outdir=None):
    
    if asciicords is not None:
        fits_file = input_filename.split(".")[0]+'.fits'
        hdu = fits.open(fits_file)
        ra = hdu[0].header['RA']
        dec = hdu[0].header['DEC']
        hdu.close()
        
        # Xmatch with the asciicoords file
        table = Table({'ra':[ra], 'dec':[dec]})
        table_coord = Table.read(asciicords, format='ascii.csv')
        column_names = ['ra', 'dec', 'ra', 'dec', 'DR3_source_id']
        xmatch_table = sky_xmatch(table, table_coord, 1800, column_names)
        source_name = xmatch_table['Name'][0]
    else:
        source_name = os.path.basename(input_filename)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    if calibration == 'flux':
        data = Table.read(input_filename, format='ascii')
        
        ax.plot(data['wavelength'], data['flux'], color='k')
        plt.title(source_name, fontsize=16, weight='bold')
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
        plt.title(source_name+', NO standard flux calibrated!', fontsize=16, weight='bold')
    
    if lines_file is not None:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        lines = Table.read(lines_file, format='ascii')
        for line, wl in zip(lines['line'], lines['wavelenght']):
            ax.axvline(wl, ls='--', alpha=0.5)
            plt.text(wl+11, 0.95, line, transform = trans, fontdict={'fontsize':12})
    
    ax.set_xlabel('Wavelength (Angstrom)', fontsize=15)
    ax.set_ylabel('Flux (number of counts)', fontsize=15)
    ax.set_xticks(np.arange(3500, 9501, 500))
    ax.set_xlim(left=3500, right=9501)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if plot:
        plt.tight_layout()
        plt.show()
        
    if outdir is not None:
        plt.savefig(f'{outdir}/{source_name}.png', bbox_inches = "tight", format = "png")
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

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
            #sigma = line_width(sig_T(T,position+dlambda), position+dlambda, R)
            #dx, y = gaussian_line(height, position+dlambda, sigma, wl_range, n_points)
            dx = np.linspace(wl_range[0], wl_range[1], n_points)
            gamma, m = 1e5, 1.6735575 * 10e-27 #PROVISIONAL
            y = voigt_line(position+dlambda, dx, gamma, T, m, R)
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
    template, using the pyasl module function puasl.crosscorrRV().

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