# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:49:06 2024

@author: Gerard Garcia
"""

from gaiaxpy import convert, calibrate
from gaiaxpy import plot_spectra
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np
from PyAstronomy import pyasl
from astropy.modeling import models
import astropy.units as u
from scipy.constants import h, c, k

def Gaia_XP(id_list, out_path=None, plot=False):
    
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
        

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

#thermal broadening
def sig_T(T, wl):
    kB = 1.381 * 10e-23 #Boltzmann constant, [J·K^{-1}]
    c = 299792458 #Speed of light, [m/s]
    mH = 1.6735575 * 10e-27 #Hydrogen mass, [kg]
    sig_T = np.sqrt(kB * T/(mH * c**2)) * wl
    return sig_T


def line_width(sig_int, wl, R):
    sig_R = wl/(2*np.sqrt(2*np.log(2))*R)
    sigma = np.sqrt(sig_int**2 + sig_R**2)
    return sigma

def gaussian_line(height, wl, sig, wl_range, n_points):
    g1 = models.Gaussian1D(height, wl, sig)
    x = np.linspace(wl_range[0], wl_range[1], n_points)
    y = g1(x)
    return x, y

def add_noise(s, SNR):
    noise = np.random.normal(0., np.mean(s)/SNR, s.shape)
    return s + noise

# Shifted line (due to radial velocity)
def rv_shift(wl, v):
    c = 299792.458 #Speed of light, [km/s]
    dlambda = wl * v/c
    return dlambda

def blackbody_radiation(wavelength, T):
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

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def rv_crosscorr_err(n_boot, wl_range, n_points, lines, line_strenght, R, T, SNR, rv=[0], skipedge=20):
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
    
    pos_RVs = [x for x in RVs if x > 0]
    neg_RVs = [x for x in RVs if x < 0]
    pos_RV_mean = np.mean(pos_RVs)
    pos_RV_std = np.std(pos_RVs)
    neg_RV_mean = np.mean(neg_RVs)
    neg_RV_std = np.std(neg_RVs)
    
    print('---')
    print(f'Positive radial velocity: {pos_RV_mean} '+u'\u00B1'+f' {pos_RV_std} km/s')
    print(f'Negative radial velocity: {neg_RV_mean} '+u'\u00B1'+f' {neg_RV_std} km/s')