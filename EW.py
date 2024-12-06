# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:29:27 2024

@author: xGeeRe
"""
from scipy.ndimage import gaussian_filter
from phot_utils import CMD
import matplotlib.pyplot as plt
from astropy.table import Table
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
import extinction


#%%
#Definie Functions

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
    emi_lines = lines[lines['line_type'] == 'emision']['line_center'].value
   
    if plot:
        su.plotter(wavelength[mask], flux[mask], figsize=(14,6), plttype='plot', ax=None,
                  xlabel=r'Wavelength [$\AA$]', ylabel='Normalized Flux', title=None,
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
    
    flux_mean = su.spec_plot(filepath, norm='region', ax=None, ylim=None, 
                             lines_file='data/spectral_lines.txt', plot=True, xmin=xmin, xmax=xmax)
    sigma=None
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
    
    line_width = float(input('Width of the line (window to fit a gaussian): '))
    gaussian_init_abs = []
    for line in abs_lines:
        sub_region = SpectralRegion((line-line_width/2)*u.angstrom, (line+line_width/2)*u.angstrom)
        sub_spectrum = extract_region(spectrum, sub_region)
        result = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(amplitude=result.amplitude.value* u.Unit('erg / (cm2 s Å)'), 
                                   mean=line*u.angstrom, stddev=result.stddev.value*u.angstrom)
        gaussian_init_abs.append(g_init)
        
    gaussian_init_emi = []
    for line in emi_lines:
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
        input_text = f'Window for the fit of absorption line {i+1} (wl axis left and right separated by a comma): '
        window = np.array((input(input_text)).split(',')).astype(float)
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
        flux_mean = su.spec_plot(filepath, norm='region', ax=None, ylim=None, 
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
        input_text = f'Window for the fit of emission line {i+1} (wl axis left and right separated by a comma): '
        window = np.array((input(input_text)).split(',')).astype(float)
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
        flux_mean = su.spec_plot(filepath, norm='region', ax=None, ylim=None, 
                                 lines_file='data/spectral_lines.txt', plot=False, xmin=xmin, xmax=xmax)
        for j, fitted_line in enumerate(flux_fit_emi):
            plt.plot(wavelength[mask], fitted_line+1* u.Unit('erg / (cm2 s Å)'))
            plt.axvspan(abs_lines[j]-ew_specutils_emi[j]/2, abs_lines[j]+ew_specutils_emi[j]/2, 
                        color='blue', alpha=0.3)
        plt.show()
        plt.close()
    
    return ew_specutils_abs, ew_quad_abs, ew_specutils_emi, ew_quad_emi
    
def NaID_extinction(ew, line='D1'):
    if line == 'D1':
        D1 = 10**(2.47*ew - 1.76)
        AVD1 = D1*3.1
        return D1, AVD1
    elif line == 'D2':
        D2 = 10**(2.16*ew - 1.91)
        AVD2 = D2*3.1
        return D2, AVD2
    
def DiBs_extinction(ew, line='5780'):
    if line == '5780':
        dib5780 = 1.978*ew +0.035
        AV5780 = dib5780*3.1
        return dib5780, AV5780
    elif line == '6614':
        dib6614 = 3.846*ew +0.072
        AV6614 = dib6614*3.1
        return dib6614, AV6614
    
def gaia_extinction(AV, Teff, how='starhorse'):
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
    
    if how == 'edr3':
        coeff = pd.read_csv('data/Fitz19_EDR3_MainSequence.csv')
        X = Teff/5040
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
        km =  a1 + a2*X + a3*X**2 + a4*X**3 + a5*AV + a6*AV**2 + a7*AV**3 + a8*AV*X + a9*AV*X**2 + a10*X*AV**2 #Gaia EDR3
        A_G = km[15]*AV
        A_BP = km[12]*AV
        A_RP = km[9]*AV
        return A_G, A_BP, A_RP
    if how == 'extinction':
        AG_ext, ABP_ext, ARP_ext = extinction.fitzpatrick99(np.array([6730, 5320, 7970]), AV, 3.1) #Extinction
        return AG_ext, ABP_ext, ARP_ext
    if how == 'starhorse':
        return AG(AV, Teff), ABP(AV, Teff), ARP(AV, Teff)
    
def mag_ext_corr(ide, AV, how):
    # Get data from the table
    table = pd.read_csv('data/extinction.csv')
    source = table[table['source_id']==int(ide)]
    phot_g_mean_mag = source['phot_g_mean_mag']
    dist = source['dist50']*1000 #parsec
    bp_rp = source['bp_rp']
    Teff = source['teff50'].iloc[0]
    
    # Compute new extinction, abs mag and BP-RP colour
    AG, ABP, ARP = gaia_extinction(AV, Teff, how)
    G = phot_g_mean_mag-5*(np.log10(dist)-1)-AG
    BP_RP = bp_rp - ABP + ARP
    
    return G, BP_RP

def new_extincted_CMD(list_ids, filepath):
    
    mg0_list = []
    bprp0_list = []
    G_na_list = []
    BP_RP_na_list = []
    G_dib_list = []
    BP_RP_dib_list = []
    for ide in list_ids:
        table = pd.read_csv('data/extinction.csv')
        source = table[table['source_id']==int(ide)]
        mg0 = source['mg0']
        bprp0 = source['bprp0']
        # NaI D
        # xrange = [5890, 5900]
        xrange = [5884, 5894]
        ews, _, _, _ = EW(filepath, xrange)
        sum_ews = sum(ews)
        _, AV = NaID_extinction(sum_ews, line='D2')
        G_na, BP_RP_na = mag_ext_corr(ide, AV, how='starhorse')
        # DiB
        # xrange = [5770, 5790]
        xrange = [6604, 6624]
        ews, _, _, _ = EW(filepath, xrange)
        sum_ews = sum(ews)
        _, AV = DiBs_extinction(sum_ews, line='6614')
        G_dib, BP_RP_dib = mag_ext_corr(ide, AV, how='starhorse')
        # Append values to lists to plot later
        mg0_list.append(mg0)
        bprp0_list.append(bprp0)
        G_na_list.append(G_na)
        BP_RP_na_list.append(BP_RP_na)
        G_dib_list.append(G_dib)
        BP_RP_dib_list.append(BP_RP_dib)
        
    CMD('data/70_targets_extended.csv', s=100, color='grey', marker='*', alpha=0.5)
    ax=plt.gca()
    ax.scatter(bprp0_list, mg0_list, s=150, color='r', marker='*')
    ax.scatter(BP_RP_na_list, G_na_list, s=150, color='blue', marker='*')
    ax.scatter(BP_RP_dib_list, G_dib_list, s=150, color='green', marker='*')
    
    ax.set_xlabel("$(BP - RP)$ [mag]", fontsize = 25)
    ax.set_ylabel("$M_{G}$ [mag]", fontsize = 25)
    ax.tick_params(labelsize = 20)   
    ax.set_xlim(-2, 1.6)
    ax.set_ylim(2, -8.5)
    plt.show()
    plt.close()
    
    
#%% Line/Spectrum Fitting

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
# Extinciton1 = 10**(2.47*(ew1.value+ew2.value) - 1.76)
Extinciton1 = 0.16*(ew1.value+ew2.value) - 0.01
Extinciton2 = 10**(2.47*(ew1_int[0]+ew2_int[0]) - 1.76)
print('---')
print('E(B-V) = ', Extinciton1,', ', Extinciton2)
print('A_V = ', Extinciton1*3.1,', ', Extinciton2*3.1)

plt.show()
plt.close()

#%% DiBs

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
Teff = 7954.713
X = Teff/5040
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
print('A_G =', A_G, ' mag; A_G_extinction =', AG_ext, ' mag; A_G_fede =', AG(A0, Teff), ' mag')
print('A_BP =', A_BP, ' mag; A_BP_extinction =', ABP_ext, ' mag; A_BP_fede =', ABP(A0, Teff), ' mag')
print('A_RP =', A_RP, ' mag; A_RP_extinction =', ARP_ext, ' mag; A_RP_fede =', ARP(A0, Teff), ' mag')

table = pd.read_csv('data/extinction.csv')
source = table[table['source_id']==508419369310190976]

phot_g_mean_mag = source['phot_g_mean_mag']
dist = source['dist50']*1000 #parsec
bp_rp = source['bp_rp']
G = phot_g_mean_mag-5*(np.log10(dist)-1)-AG(A0, Teff)
BP_RP = bp_rp - ABP(A0, Teff) + ARP(A0, Teff)


print('G = ', G.iloc[0], ' mag')
print('BP-RP = ', BP_RP.iloc[0], ' mag')

print('---')
print('DiB (5780Å)')

A0 = 2.709994885071724
A_G = km[15]*A0
A_BP = km[12]*A0
A_RP = km[9]*A0
AG_ext, ABP_ext, ARP_ext = extinction.fitzpatrick99(np.array([6730, 5320, 7970]), A0, 3.1)
print('A_G =', A_G, ' mag; A_G_extinction =', AG_ext, ' mag; A_G_fede =', AG(A0, Teff), ' mag')
print('A_BP =', A_BP, ' mag; A_BP_extinction =', ABP_ext, ' mag; A_BP_fede =', ABP(A0, Teff), ' mag')
print('A_RP =', A_RP, ' mag; A_RP_extinction =', ARP_ext, ' mag; A_RP_fede =', ARP(A0, Teff), ' mag')
G = phot_g_mean_mag-5*(np.log10(dist)-1)-AG(A0, Teff)
BP_RP = bp_rp - ABP(A0, Teff) + ARP(A0, Teff)
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
