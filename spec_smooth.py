# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:30:30 2024

@author: xGeeRe
"""

import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from specutils import Spectrum1D
from specutils.manipulation import gaussian_smooth
import matplotlib.pyplot as plt
import astropy.units as u
import os
import pandas as pd
import spectra_utils as su
from scipy.interpolate import interp1d
from astropy.table import Table

name = '2200433413577635840'
directory = 'data/FIES-M_spectra'
files = os.listdir(directory)
spec = pd.read_csv(f'{directory}/{name}.txt', sep=' ')
wavelengths = np.array(spec['wavelength'])
flux = np.array(spec['flux']/np.mean(spec['flux']))

# Example wavelength array (in Angstroms) and synthetic flux with some noise
# wavelengths = np.linspace(4000, 5000, 1000)  # Wavelengths from 4000 to 5000 Å
# flux = np.sin(wavelengths / 100) + np.random.normal(0, 0.1, len(wavelengths))  # Flux with noise

# Create a Spectrum1D object
spectrum = Spectrum1D(flux=flux* u.m / u.m, spectral_axis=wavelengths * u.AA)

# Initial and final resolutions
R_final = 2475
low_res_pixel_scale = 1.04
# R_final = 270
# low_res_pixel_scale = 4.6

# Kernel width in wavelength units (FWHM of the broadening)
# We convert from FWHM to standard deviation using FWHM = 2.355 * sigma
fwhm_lambda = np.mean(wavelengths) / R_final
# fwhm_lambda = 6560 / R_final
sigma_lambda = fwhm_lambda / 2.355

# Define a Gaussian kernel based on the calculated sigma (in pixels)
# The Gaussian1DKernel expects the standard deviation in pixels, 
# so we need to convert the sigma_lambda into pixel units
pixel_scale = np.mean(np.diff(wavelengths))  # Wavelength step size in Å
sigma_pixels = sigma_lambda / pixel_scale

# Create the Gaussian kernel
gaussian_kernel = Gaussian1DKernel(sigma_pixels)

# Smooth the spectrum
smoothed_flux = convolve(spectrum.flux, gaussian_kernel)

# Resample to new pixel scale
new_wavelengths = np.arange(wavelengths[0], wavelengths[-1], low_res_pixel_scale)
interpolator = interp1d(wavelengths, smoothed_flux, kind='linear', bounds_error=False, fill_value="extrapolate")
new_flux = interpolator(new_wavelengths)

# Bin new (low-res) spectrum

new_wavelength_bins = np.arange(wavelengths[0], wavelengths[-1], low_res_pixel_scale)
binned_wavelengths = []
binned_flux = []

for i in range(len(new_wavelength_bins) - 1):
    # Find the indices of the wavelengths that fall within the current bin
    indices_in_bin = np.where((wavelengths >= new_wavelength_bins[i]) & (wavelengths < new_wavelength_bins[i + 1]))[0]
    
    if len(indices_in_bin) > 0:
        # Calculate the average wavelength and flux for this bin
        avg_wavelength = np.mean(wavelengths[indices_in_bin])
        avg_flux = np.mean(smoothed_flux[indices_in_bin])
        
        binned_wavelengths.append(avg_wavelength)
        binned_flux.append(avg_flux)

# Convert lists to numpy arrays
binned_wavelengths = np.array(binned_wavelengths)
binned_flux = np.array(binned_flux)

# Plot the original and smoothed spectrum
fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(wavelengths, flux, label="Original Spectrum (R=25000, FIES low-res)", color='blue')
# ax.plot(wavelengths, smoothed_flux, label=f"Smoothed Spectrum (R={R_final})", color="red", lw=2)
# ax.scatter(new_wavelengths, new_flux, label=f"Resampled Spectrum (R={R_final})", color="orange", zorder=5)
# ax.scatter(binned_wavelengths, binned_flux, label="Binned Spectrum (R≈2000)", color='lightgreen', zorder=5)
su.spec_velocity(6562.8, wavelengths, flux, legend=False,
                  color='b', ax=ax)
# su.spec_velocity(6562.8, wavelengths, smoothed_flux, legend=False,
#                   color='r', ax=ax)
# su.spec_velocity(6562.8, new_wavelengths, new_flux, legend=False,
#                   color='orange', ax=ax, line='scatter', zorder=5)
su.spec_velocity(6562.8, binned_wavelengths, binned_flux, legend=False,
                  color='r', ax=ax, line='line')

# spectrum = Table.read('data/cafos_spectra/spectra1D_dswfz_uniB_0208.txt', format='ascii')
# mask = (spectrum['wavelength']>6500)*(spectrum['wavelength']<6620)

# su.spec_velocity(6562.8, spectrum['wavelength'][mask], spectrum['flux'][mask], legend=False,
#                  color='k', ax=ax, line='scatter', zorder=5)

# plt.xlabel("Wavelength (Å)", fontsize=14)
# plt.ylabel("Normalized Flux", fontsize=14)
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlim(left=6550, right=6570)
# plt.ylim(bottom=0.7, top=1.4)
plt.xlim(left=-500, right=500)
plt.ylim(bottom=0.7, top=1.4)
# plt.legend(["Original Spectrum (R=25000, FIES low-res)", f"Smoothed Spectrum (R={R_final})", 'Resampling', f"Binned Spectrum (R={R_final})"], fontsize=13)
plt.legend(["Original Spectrum (R=25000)", f"Smoothed Spectrum (R={R_final})"], fontsize=15)
plt.tight_layout()
# plt.show()
plt.savefig('GTC_fig2.pdf', format='pdf')

#%%

import matplotlib.pyplot as plt
import numpy as np

# Your list of magnitudes
magnitudes = [
    9.786373, 15.609404, 12.809022, 14.231659, 13.650927, 14.976458, 
    13.825504, 13.552832, 15.639525, 13.922831, 16.331625, 14.891698, 
    15.118116, 16.48481, 8.571921, 13.291793, 15.344664, 15.61235, 
    15.3971815, 15.594354, 10.358919, 13.273162, 16.50223, 15.148152, 
    11.264488, 14.061699, 9.597249, 9.714202, 11.6453, 12.706895, 
    10.971821, 15.71619, 11.252727
]

# Plot histogram with bins and range starting at 9.5
plt.figure(figsize=(10, 8))
plt.hist(magnitudes, bins=14, range=(9.5, max(magnitudes)), edgecolor='black', alpha=0.75)

# Set the x-axis ticks to be every 0.5 starting from 9.5
plt.xticks(np.arange(9.5, max(magnitudes) + 0.5, 0.5), size=14)

# Add titles and labels
plt.title("Magnitude Distribution", fontsize=14)
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("#", fontsize=14)
plt.yticks(size=14)

# Show grid and make layout tight
plt.grid(zorder=-1)
plt.tight_layout()

# Show the plot
plt.show()




#%%

import phot_utils as pu

# id_list=[2002117151282819840,
# 2007318661608788096,
# 2006912396372680960,
# 2006088484204609408,
# 2173852964799716480,
# 2175699216614191360,
# 2169083008475385856,
# 2163542397602876800,
# 2083649030845658624,
# 2054338249889402880,
# 2061252975440642816,
# 2027563492489195520,
# 4321276689423536384,
# 4515124540765147776,
# 4263591911398361472,
# 4519475166529738112,
# 4281886474885416064,
# 4260141158544875008,
# 4272588356022299520,
# 4271992661242707200,
# 4096527235637366912,
# 2931553674771048704,
# 2934216142176785920,
# 3355776901779440384,
# 3369399099232812160,
# 473575777103322496,
# 461193695624775424,
# 508419369310190976]

id_list=[5328449200388495616,
5323384162646755712,
5868425648663616768,
6228685649971375616,
5866345647515558400,
5338183383022960512,
5311969857556479616,
5965503866703572480,
5880159842877908352,
6053890788968694656,
5617186348318629248,
5524022735225482624,
5962956195185292288,
4094491141885400576,
5866474526572151936,
5593826360487373696,
5882737819707242240,
5350869719969619840,
5599309216965305728,
4054010697162430592
]
table = pd.read_csv('data/70_targets_extended.csv')
# mask = table['source_id']==table1
filtered_df = table[table['source_id'].isin(id_list)]
fig, ax = plt.subplots(figsize=(8,8))
pu.CMD('data/70_targets_extended.csv', s=50, color='k', alpha=0.5, ax=ax)
ax.scatter(filtered_df['bprp0'], filtered_df['mg0'], s=300, color='r', marker='*')
ax.set_xlabel("$(BP - RP)$ [mag]", fontsize = 16)
ax.set_ylabel("$M_{G}$ [mag]", fontsize = 16)
ax.tick_params(labelsize = 16)

# plt.show()
plt.savefig('GTC_fig1.pdf', format='pdf')