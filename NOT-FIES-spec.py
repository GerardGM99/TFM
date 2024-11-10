# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:28:32 2024

@author: xGeeRe
"""

import spectra_utils as su
import pandas as pd
import os
from astropy.io import fits

path28 = '203/2024-10-28/fies/reduced/'
files28 = os.listdir(path28)
plotdir = os.path.join(path28,'plots')
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

for file in files28:
    if 'merge.fits' in file:
        header = fits.getheader(os.path.join(path28, file))
        RA=header['RA']
        DEC=header['DEC']
        obs_time=header['DATE']
        obj_name=header['OBJECT']
        
        spectrum = pd.read_csv(os.path.join(path28, file.replace('fits', 'txt')), sep=' ')
        
        wavelengths = spectrum['wavelength']
        fluxes = spectrum['flux']


        su.classification_grid(wavelengths, fluxes, obj_name+os.path.basename(file.replace('fits', 'txt')), 
                               site='lapalma', RA=RA, DEC=DEC, obs_time=obs_time,
                               savepath=plotdir)


path29 = '203/2024-10-29/fies/reduced/'
files29 = os.listdir(path29)
plotdir = os.path.join(path29,'plots')
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

for file in files29:
    if 'merge.fits' in file:
        header = fits.getheader(os.path.join(path29, file))
        RA=header['RA']
        DEC=header['DEC']
        obs_time=header['DATE']
        obj_name=header['OBJECT']
        
        spectrum = pd.read_csv(os.path.join(path29, file.replace('fits', 'txt')), sep=' ')
        
        wavelengths = spectrum['wavelength']
        fluxes = spectrum['flux']


        su.classification_grid(wavelengths, fluxes, obj_name+os.path.basename(file.replace('fits', 'txt')), 
                               site='lapalma', RA=RA, DEC=DEC, obs_time=obs_time,
                               savepath=plotdir)
        
        
#%%ALFOSC

import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pandas as pd
import spectra_utils as su

path = '203/2024-10-29/alfosc/not_alfosc_L/Science/'
file = 'spec1d_ALHj290141-at2024znz_ALFOSC_20241030T062521.487.fits'
hdu = fits.open(os.path.join(path, file))

for i in range(1,5):
    head = hdu[i].header
    
    data = hdu[i].data
    
    wavelength = data['OPT_WAVE']   # Optimal wavelength
    # flux = data['OPT_COUNTS']       # Optimal flux counts
    # flux_error = data['OPT_COUNTS_SIG']
    flux = data['OPT_FLAM']       # Optimal flux 
    flux_error = data['OPT_FLAM_SIG']
    mask = (wavelength>3800)&(wavelength<9500)
    
    # Plot the spectrum with error bars
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength[mask], flux[mask], color='blue', lw=1, label='Flux')
    plt.xlabel('Wavelength (Å)', fontsize=16, fontfamily='serif')
    plt.ylabel(r'Flux (1e-17 erg/s/$cm^2$/Å)', fontsize=16, fontfamily='serif')
    name = file.split('_')[1].split('-')[1]
    plt.title(f'1D Spectrum for {name}, i: {i}', fontsize=18, fontfamily='serif', weight='bold')
    plt.minorticks_on()
    plt.tick_params(which='both', labelsize=15, direction='in')
    plt.minorticks_on()
    plt.ylim(bottom=-4)
    plt.tight_layout()
    plt.show()
    
    # if i==4:
    #     df = pd.DataFrame({'wavelength':wavelength, 'flux':flux})
    #     out = f'data/FIES-M_spectra/{name}'
    #     df.to_csv(f'{out}.txt', sep=' ', index=False)