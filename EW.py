# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:29:27 2024

@author: xGeeRe
"""

#Definie Functions

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def find_lines(wavelength, flux, xmin, xmax, threshold=1, noise_range=None, plot=False):
   
    mask = (wavelength>xmin)&(wavelength<xmax)
   
    # Normalize the flux using 60% of the range given (30% from the redder part and 30% form the bluer part)
    length = xmax-xmin
    mask_blue = (wavelength > xmin) & (wavelength < (xmin+length*0.3))
    flux_blue = flux[mask_blue]
    mask_red = (wavelength < xmax) & (wavelength > (xmax-length*0.3))
    flux_red = flux[mask_red]
    flux_mean = (np.mean(flux_blue)+np.mean(flux_red))/2
    flux = flux/flux_mean -1
   
    # Create specutils Spectrum1D class
    spectrum = Spectrum1D(spectral_axis=wavelength[mask] * u.angstrom, flux=flux[mask] * u.Unit('erg / (cm2 s Å)') )
   
    # Find lines using specutils find_lines_derivative or find_lines_threshold
    if noise_range is None:
        lines = find_lines_derivative(spectrum, flux_threshold=threshold)
    else:
        noise_region = SpectralRegion(noise_range[0]*u.angstrom, noise_range[1]*u.angstrom)
        spectrum = noise_region_uncertainty(spectrum, noise_region)
        lines = find_lines_threshold(spectrum, noise_factor=threshold)
   
    abs_lines = lines[lines['line_type'] == 'absorption']['line_center'].value
    emi_lines = lines[lines['line_type'] == 'emision']['line_center'].value
   
    if plot:
        su.plotter(wavelength, flux, figsize=(14,6), plttype='plot', ax=None,
                  xlabel=r'Wavelength [$\AA$]', ylabel='Normalized Flux', title=None,
                  xmin=xmin, xmax=xmax, ylim=None, xinvert=False, yinvert=False, legend=False,
                  show=False, savepath=None, saveformat='png', color='k')
        ax=plt.gca()
        for line in abs_lines:
            ax.axvline(line, color='purple', alpha=0.6)
        for line in emi_lines:
            ax.axvline(line, color='g', alpha=0.6)
        plt.tight_layout()
        plt.show()
        plt.close()
   
    return abs_lines, emi_lines

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
print('A_V = ', Extinciton1*3.1,', ', Extinciton2*3.1)

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
import numpy as np
import extinction

# Friedrich Anders (https://github.com/fjaellet/gaia_edr3_photutils/tree/main), Schlafly+2016 extinction law
def AG(AV, Teff):
    """
    Compute A_G as a function of AV (a.k.a. A_0) and Teff (using pysvo)
    
    Takes:
        AV   (array/float) - extinction at lambda=5420 AA in mag
        Teff (array/float) - effective temperature in K
    Returns:
        AG   (array/float) - extinction in the Gaia EDR3 G band
    """
    coeffs = np.array([[ 7.17833016e-01, -1.88633321e-02,  5.77430628e-04],
       [ 2.84726306e-05, -1.65986478e-06,  3.29897761e-08],
       [-4.70938509e-10,  2.76393659e-11, -5.56454892e-13]])
    return np.polynomial.polynomial.polyval2d(Teff, AV, coeffs) * AV

def ARP(AV, Teff):
    """
    Compute A_RP as a function of AV (a.k.a. A0) and Teff (using pysvo)
    
    Takes:
        AV   (array/float) - extinction at lambda=5420 AA in mag
        Teff (array/float) - effective temperature in K
    Returns:
        ARP   (array/float) - extinction in the Gaia EDR3 G_RP band
    """
    coeffs = np.array([[ 5.87378504e-01, -6.37597056e-03,  8.71180628e-05],
       [ 4.71862901e-06, -7.42096958e-09, -4.51905872e-09],
       [-7.99119123e-11,  2.80843682e-13,  7.23076354e-14]])
    return np.polynomial.polynomial.polyval2d(Teff, AV, coeffs) * AV

def ABP(AV, Teff):
    """
    Compute A_BP as a function of AV (a.k.a. A0) and Teff (using pysvo)
    
    Takes:
        AV   (array/float) - extinction at lambda=5420 AA in mag
        Teff (array/float) - effective temperature in K
    Returns:
        ARP   (array/float) - extinction in the Gaia EDR3 G_BP band
    """
    coeffs = np.array([[ 9.59835295e-01, -1.16380247e-02,  3.50836411e-04],
       [ 1.82122771e-05, -9.17453966e-07,  1.43568978e-08],
       [-2.90443152e-10,  1.41611091e-11, -2.10356011e-13]])
    return np.polynomial.polynomial.polyval2d(Teff, AV, coeffs) * AV


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
km =  a1 + a2*X + a3*X**2 + a4*X**3 + a5*A0 + a6*A0**2 + a7*A0**3 + a8*A0*X + a9*A0*X**2 + a10*X*A0**2 #Gaia EDR3
# print(km)
AG_ext, ABP_ext, ARP_ext = extinction.fitzpatrick99(np.array([6730, 5320, 7970]), A0, 3.1) #Extinction

print('NaI D')
A_G = km[15]*A0
A_BP = km[12]*A0
A_RP = km[9]*A0
print('A_G =', A_G, ' mag; A_G_extinction =', AG_ext, ' mag; A_G_fede =', AG(A0, X), ' mag')
print('A_BP =', A_BP, ' mag; A_BP_extinction =', ABP_ext, ' mag; A_BP_fede =', ABP(A0, X), ' mag')
print('A_RP =', A_RP, ' mag; A_RP_extinction =', ARP_ext, ' mag; A_RP_fede =', ARP(A0, X), ' mag')

table = pd.read_csv('data/extinction.csv')
source = table[table['source_id']==508419369310190976]

phot_g_mean_mag = source['phot_g_mean_mag']
dist = source['dist50']*1000 #parsec
bp_rp = source['bp_rp']
G = phot_g_mean_mag-5*(np.log10(dist)-1)-AG(A0, X)
BP_RP = bp_rp - ABP(A0, X) + ARP(A0, X)


print('G = ', G.iloc[0], ' mag')
print('BP-RP = ', BP_RP.iloc[0], ' mag')

print('---')
print('DiB (5780Å)')

A0 = 2.709994885071724
A_G = km[15]*A0
A_BP = km[12]*A0
A_RP = km[9]*A0
AG_ext, ABP_ext, ARP_ext = extinction.fitzpatrick99(np.array([6730, 5320, 7970]), A0, 3.1)
print('A_G =', A_G, ' mag; A_G_extinction =', AG_ext, ' mag; A_G_fede =', AG(A0, X), ' mag')
print('A_BP =', A_BP, ' mag; A_BP_extinction =', ABP_ext, ' mag; A_BP_fede =', ABP(A0, X), ' mag')
print('A_RP =', A_RP, ' mag; A_RP_extinction =', ARP_ext, ' mag; A_RP_fede =', ARP(A0, X), ' mag')
G = phot_g_mean_mag-5*(np.log10(dist)-1)-AG(A0, X)
BP_RP = bp_rp - ABP(A0, X) + ARP(A0, X)
print('G = ', G.iloc[0], ' mag')
print('BP-RP = ', BP_RP.iloc[0], ' mag')

print('---')
print('Original Starhorse values')
print('G = ', source['mg0'].iloc[0], ' mag')
print('BP-RP = ', source['bprp0'].iloc[0], ' mag')

#%%

from phot_utils import CMD
import matplotlib.pyplot as plt
from astropy.table import Table

# t = Table.read("tables/starhorse2-result.vot", format="votable")
# CMD('data/starhorse2-result.vot', table_format="votable", s=10, density=True, 
#     cmap='viridis', norm='log')
CMD('data/70_targets_extended.csv', s=100, color='grey', marker='*', alpha=0.5)
ax=plt.gca()
ax.scatter(-0.7655574331391879, -5.131191698585232, s=200, color='blue', marker='*')
ax.scatter(0.09557456, -3.6793861 , s=200, color='r', marker='*')
ax.scatter(0.018795921067792154, -3.765402096655256 , s=200, color='green', marker='*')
ax.set_xlabel("$(BP - RP)$ [mag]", fontsize = 25)
ax.set_ylabel("$M_{G}$ [mag]", fontsize = 25)
ax.tick_params(labelsize = 20)
ax.set_xlim(-1, 1.5)
# ax.set_ylim(2, -7)
plt.show()
plt.close()
