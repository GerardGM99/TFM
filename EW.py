# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:29:27 2024

@author: xGeeRe
"""

from specutils import Spectrum1D
from specutils.analysis import equivalent_width
from specutils import SpectralRegion
import astropy.units as u
import numpy as np
import pandas as pd
import os

path = '203/FIES/28/'
file = 'FIHj280108_step011_merge.txt'

# path = '203/ALFOSC/'
# file = '187219239343050880_037.txt'

# Step 1: Load the spectrum data
# Replace with your actual data file
spectrum = pd.read_csv(os.path.join(path,file), sep=' ')
wavelength = np.array(spectrum['wavelength']) * u.angstrom
flux = np.array(1e-17 * spectrum['flux']) * u.Unit('erg / (cm2 s Å)')  # Replace with appropriate flux unit

# Step 2: Create a Spectrum1D object
spectrum = Spectrum1D(spectral_axis=wavelength, flux=flux)

# Step 3: Define the region of interest (ROI)

roi = SpectralRegion(5894 * u.angstrom, 5896.5 * u.angstrom)  # Example region for DiB

# Step 4: Compute the equivalent width
ew = equivalent_width(spectrum, regions=roi)

print(f"Equivalent Width: {ew:.2f}")


#%%

import numpy as np
from scipy.integrate import simps

# Load wavelength and flux data
# Replace 'your_data_file' with the actual file path
spectrum = pd.read_csv('203/ALFOSC/187219239343050880_037.txt', sep=' ')
wavelength = np.array(spectrum['wavelength']) * u.angstrom
flux = np.array(1e-17 * spectrum['flux']) * u.Unit('erg / (cm2 s Å)')  # Replace with appropriate flux unit


# Step 1: Normalize flux to continuum (assume pre-determined or smooth baseline fitting)
continuum_level = 1e-17 * spectrum['flux'][((4950<spectrum['wavelength'])&(spectrum['wavelength']<5000))|((5010<spectrum['wavelength'])&(spectrum['wavelength']<5060))]  
mean_cont = np.mean(continuum_level)
normalized_flux = flux / mean_cont

# Step 2: Define the region of interest (ROI) around the absorption/emission line
line_min, line_max = 5765* u.angstrom, 5792* u.angstrom  # Example wavelength range for DiB
mask = (wavelength >= line_min) & (wavelength <= line_max)
wavelength_roi = wavelength[mask]
flux_roi = normalized_flux[mask]

# Step 3: Compute the equivalent width (EW) using numerical integration
ew = simps(1* u.Unit('erg / (cm2 s Å)') - flux_roi, x=wavelength_roi)

print(f"Equivalent Width: {ew:.2f} Å")

#%%

import matplotlib.pyplot as plt
import spectra_utils as su
import os
import pandas as pd
# from astropy.table import Table
from scipy.stats import norm

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# path = '203/ALFOSC/'
# file = '187219239343050880_037.txt'

path = '203/FIES/28/'
file = 'FIHj280108_step011_merge.txt'
spectrum = pd.read_csv(os.path.join(path,file), sep=' ')
wl = spectrum['wavelength']
flux = spectrum['flux']

# asciicords='data/TFM_table.csv'
# table_coord = Table.read(asciicords, format='ascii.csv')
# name = file.split('.')[0].split('_')[0]
# Av = table_coord['$A_V$'][table_coord['Gaia source ID']==int(name)]

flux_mean = su.spec_plot(os.path.join(path,file), norm='region', ax=None, ylim=None, lines_file='data/spectral_lines.txt', plot=False, xmin=5890, xmax=5900)

x = np.linspace(5892, 5898, 3000)
mask_gaus_red = (wl>5895.1)&(wl<5898)
mu, sigma = norm.fit(flux[mask_gaus_red]/flux_mean)
plt.plot(x, 1+gaussian(x-5896, -1, mu, sigma), color='r', label='Gaussian fit')
# mu2, sigma2 = norm.fit(a2063['v_pec'][(a2063['v_pec']>sigma_i(0.040,0.034))&(a2063['v_pec']<4000)])
plt.axhline(1)
plt.axvline(5895.1)
# plt.axvspan(5895.5-2.51/2, 5895.5+2.51/2, color='blue', alpha=0.3)
plt.show()
plt.close()
# out_name = os.path.join(plotdir, file.split('.')[0])
# plt.savefig(f'{out_name}.png', bbox_inches = "tight", format = "png")
# plt.close()

#%% Line/Spectrum Fitting

import matplotlib.pyplot as plt
import spectra_utils as su
import os
import pandas as pd
import numpy as np
from astropy.modeling import models
import astropy.units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_lines
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.analysis import equivalent_width
from scipy.integrate import quad

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Read spectrum
path = '203/FIES/28/'
file = 'FIHj280104_step011_merge.txt'
spectrum = pd.read_csv(os.path.join(path,file), sep=' ')
wavelength = np.array(spectrum['wavelength'])
flux = np.array(spectrum['flux'])
xmin=5890
xmax=5900
mask = (wavelength>xmin)&(wavelength<xmax)

length = xmax-xmin
mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.4))
flux_blue = flux[mask_blue]
mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.4))
flux_red = flux[mask_red]
flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
flux = flux/flux_mean -1

spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') )

# region = SpectralRegion(xmin*u.angstrom, xmax*u.angstrom)

noise_region = SpectralRegion(5891*u.angstrom, 5894*u.angstrom)
# spectrum = noise_region_uncertainty(spectrum, noise_region)
# lines = find_lines_threshold(spectrum, noise_factor=1)
lines = find_lines_derivative(spectrum, flux_threshold=0.4)

flux_mean = su.spec_plot(os.path.join(path,file), norm='region', ax=None, ylim=None, lines_file='data/spectral_lines.txt', plot=False, xmin=xmin, xmax=xmax)

abs_lines = []
amplitude = []
stddev = []
for i, line in enumerate(lines[lines['line_type'] == 'absorption']['line_center'].value):
    plt.axvline(line, color='r', alpha=0.6)
    sub_region = SpectralRegion((line-0.4)*u.angstrom, (line+0.4)*u.angstrom)
    sub_spectrum = extract_region(spectrum, sub_region)
    result = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
    abs_lines.append(line)
    amplitude.append(result.amplitude.value)
    stddev.append(result.stddev.value)

g1_init = models.Gaussian1D(amplitude=amplitude[0]* u.Unit('erg / (cm2 s Å)'), mean=abs_lines[0]*u.angstrom, stddev=stddev[0]*u.angstrom)
g2_init = models.Gaussian1D(amplitude=amplitude[1]* u.Unit('erg / (cm2 s Å)'), mean=abs_lines[1]*u.angstrom, stddev=stddev[1]*u.angstrom)
# g3_init = models.Gaussian1D(amplitude=amplitude[2]* u.Unit('erg / (cm2 s Å)'), mean=abs_lines[2]*u.angstrom, stddev=stddev[2]*u.angstrom)

# g123_fit = fit_lines(spectrum, g1_init+g2_init)
# y_fit = g123_fit(wavelength[mask]*u.angstrom)
# plt.plot(wavelength[mask], y_fit+1* u.Unit('erg / (cm2 s Å)'))

g1_fit= fit_lines(spectrum, g1_init, window=(5894*u.angstrom,5895*u.angstrom))
y1_fit = g1_fit(wavelength[mask]*u.angstrom)

spectrum2 = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') - y1_fit)
g2_fit= fit_lines(spectrum2, g2_init, window=(5894*u.angstrom,5897*u.angstrom))
y2_fit = g2_fit(wavelength[mask]*u.angstrom)
# g3_fit= fit_lines(spectrum, g2_init, window=(5895.59*u.angstrom,5897*u.angstrom))
# y3_fit = g3_fit(wavelength[mask]*u.angstrom)
plt.plot(wavelength[mask], y1_fit+1* u.Unit('erg / (cm2 s Å)'))
plt.plot(wavelength[mask], y2_fit+1* u.Unit('erg / (cm2 s Å)'))
# plt.plot(wavelength[mask], y3_fit+1* u.Unit('erg / (cm2 s Å)'))

ew1 = equivalent_width(spectrum+1* u.Unit('erg / (cm2 s Å)'), regions=SpectralRegion(5894*u.angstrom, 5895*u.angstrom))
ew2 = equivalent_width(spectrum2+1* u.Unit('erg / (cm2 s Å)'), regions=SpectralRegion(5894*u.angstrom, 5897*u.angstrom))
plt.axvspan(abs_lines[0]-ew1.value/2, abs_lines[0]+ew1.value/2, color='blue', alpha=0.3)
plt.axvspan(abs_lines[1]-ew2.value/2, abs_lines[1]+ew2.value/2, color='blue', alpha=0.3)

ew1_int = quad(gaussian, args=(-g1_fit.amplitude.value, g1_fit.mean.value, g1_fit.stddev.value), a=5894, b=5898)
print('EW1: ', ew1.value, 'Å, ', ew1_int[0], 'Å')
ew2_int = quad(gaussian, args=(-g2_fit.amplitude.value, g2_fit.mean.value, g2_fit.stddev.value), a=5894, b=5898)
print('EW2: ', ew2.value, 'Å, ', ew2_int[0], 'Å')

# For the D1 line (5896 Å)
Extinciton1 = 10**(2.47*(ew1.value+ew2.value) - 1.76)
Extinciton2 = 10**(2.47*(ew1_int[0]+ew2_int[0]) - 1.76)
print('---')
print('E(B-V) = ', Extinciton1,', ', Extinciton2)
print('A_V = ', Extinciton1*3.1, Extinciton2*3.1)

plt.show()
plt.close()

#%% DiBs

import matplotlib.pyplot as plt
import spectra_utils as su
import os
import pandas as pd
import numpy as np
from astropy.modeling import models
import astropy.units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_lines
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.analysis import equivalent_width
from scipy.integrate import quad

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Read spectrum
path = '203/FIES/28/'
file = 'FIHj280104_step011_merge.txt'
spectrum = pd.read_csv(os.path.join(path,file), sep=' ')
wavelength = np.array(spectrum['wavelength'])
flux = np.array(spectrum['flux'])
xmin=5770
xmax=5790
mask = (wavelength>xmin)&(wavelength<xmax)

length = xmax-xmin
mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.4))
flux_blue = flux[mask_blue]
mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.4))
flux_red = flux[mask_red]
flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
flux = flux/flux_mean -1

spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') )

# region = SpectralRegion(xmin*u.angstrom, xmax*u.angstrom)

noise_region = SpectralRegion(5770*u.angstrom, 5776*u.angstrom)
# spectrum = noise_region_uncertainty(spectrum, noise_region)
# lines = find_lines_threshold(spectrum, noise_factor=3)
lines = find_lines_derivative(spectrum, flux_threshold=0.4)

flux_mean = su.spec_plot(os.path.join(path,file), norm='region', ax=None, ylim=None, lines_file='data/spectral_lines.txt', plot=False, xmin=xmin, xmax=xmax)

abs_lines = []
amplitude = []
stddev = []
# for i, line in enumerate(lines[lines['line_type'] == 'absorption']['line_center'].value):
for i, line in enumerate([5780]):
    # plt.axvline(line, color='r', alpha=0.6)
    sub_region = SpectralRegion((line-2.5)*u.angstrom, (line+2.5)*u.angstrom)
    sub_spectrum = extract_region(spectrum, sub_region)
    result = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
    abs_lines.append(line)
    amplitude.append(result.amplitude.value)
    stddev.append(result.stddev.value)

g1_init = models.Gaussian1D(amplitude=amplitude[0]* u.Unit('erg / (cm2 s Å)'), mean=abs_lines[0]*u.angstrom, stddev=stddev[0]*u.angstrom)
# g2_init = models.Gaussian1D(amplitude=amplitude[1]* u.Unit('erg / (cm2 s Å)'), mean=abs_lines[1]*u.angstrom, stddev=stddev[1]*u.angstrom)

g1_fit= fit_lines(spectrum, g1_init, window=(5775*u.angstrom,5783*u.angstrom))
y1_fit = g1_fit(wavelength[mask]*u.angstrom)

# spectrum2 = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') - y1_fit)
# g2_fit= fit_lines(spectrum2, g2_init, window=(5894*u.angstrom,5897*u.angstrom))
# y2_fit = g2_fit(wavelength[mask]*u.angstrom)
# # g3_fit= fit_lines(spectrum, g2_init, window=(5895.59*u.angstrom,5897*u.angstrom))
# # y3_fit = g3_fit(wavelength[mask]*u.angstrom)

plt.plot(wavelength[mask], y1_fit+1* u.Unit('erg / (cm2 s Å)'))
# plt.plot(wavelength[mask], y2_fit+1* u.Unit('erg / (cm2 s Å)'))

ew1 = equivalent_width(spectrum+1* u.Unit('erg / (cm2 s Å)'), regions=SpectralRegion(5775*u.angstrom, 5785*u.angstrom))
# ew2 = equivalent_width(spectrum2+1* u.Unit('erg / (cm2 s Å)'), regions=SpectralRegion(5894*u.angstrom, 5897*u.angstrom))
plt.axvspan(abs_lines[0]-ew1.value/2, abs_lines[0]+ew1.value/2, color='blue', alpha=0.3)
# plt.axvspan(abs_lines[1]-ew2.value/2, abs_lines[1]+ew2.value/2, color='blue', alpha=0.3)

ew1_int = quad(gaussian, args=(-g1_fit.amplitude.value, g1_fit.mean.value, g1_fit.stddev.value), a=5775, b=5785)
print(ew1.value, ew1_int[0])

# For the DiBs line at 5780 Å
Extinciton1 = 1.978*ew1.value - 0.035
Extinciton2 = 1.978*ew1_int[0] - 0.035
print('---')
print('E(B-V) = ', Extinciton1,', ', Extinciton2)
print('A_V = ', Extinciton1*3.1, Extinciton2*3.1)

plt.show()
plt.close()

#%%

import pandas as pd

coeff = pd.read_csv('data/Fitz19_EDR3_MainSequence.csv')
X = 7954.713/5040
A0 = 4.989957343355702
a1 = coeff['Intercept']
a2 = coeff['X']
a3 = coeff['X2']
a4 = coeff['X3']
a5 = coeff['A']
a6 = coeff['A2']
a7 = coeff['A3']
a8 = coeff['XA']
a9 = coeff['AX2']
a10 = coeff['XA2']
km =  a1 + a2*X + a3*X**2 + a4*X**3 + a5*A0 + a6*A0**2 + a7*A0**3 + a8*A0*X + a9*A0*X**2 + a10*X*A0**2
# print(km)
print('NaI D')
A_G = km[15]*A0
A_BP = km[12]*A0
A_RP = km[9]*A0
print('A_G =', A_G, ' mag')
print('A_BP =', A_BP, ' mag')
print('A_RP =', A_RP, ' mag')

table = pd.read_csv('data/extinction.csv')
source = table[table['source_id']==508419369310190976]

phot_g_mean_mag = source['phot_g_mean_mag']
dist = source['dist50']*1000 #parsec
bp_rp = source['bp_rp']
G = phot_g_mean_mag-5*(np.log10(dist)-1)-A_G
BP_RP = bp_rp - A_BP + A_RP


print('G = ', G.iloc[0], ' mag')
print('BP-RP = ', BP_RP.iloc[0], ' mag')

print('---')
print('DiB (5780Å)')

A0 = 2.709994885071724
A_G = km[15]*A0
A_BP = km[12]*A0
A_RP = km[9]*A0
print('A_G =', A_G, ' mag')
print('A_BP =', A_BP, ' mag')
print('A_RP =', A_RP, ' mag')
G = phot_g_mean_mag-5*(np.log10(dist)-1)-A_G
BP_RP = bp_rp - A_BP + A_RP
print('G = ', G.iloc[0], ' mag')
print('BP-RP = ', BP_RP.iloc[0], ' mag')

print('---')
print('Original Starhorse values')
print('G = -3.67939 mag')
print('BP-RP = 0.9948 mag')