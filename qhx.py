# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:33:13 2024

@author: xGeeRe
"""

from QhX import plots
from QhX.utils import *
from QhX.algorithms.wavelets.wwtz import *
from QhX.calculation import *
import warnings
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import os, sys
# from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from lightcurve_utils import lc_folding, lc_combined, bulk_combine



warnings.simplefilter("error", OptimizeWarning)

# period = 100  # days
# amplitude = 0.3
# tt, yy = simple_mock_lc(time_interval=10, num_points=1000, frequency=period, amplitude=amplitude, percent=0.5, magnitude=22)
# plots.fig_plot(tt, yy)

table = pd.read_csv('data/mega_xmatch.csv')

# mask = table['Period(me)'] == '-'
mask = table['Gaia source ID_1'] == 187219239343050880

for star in table['Gaia source ID_1'][mask]:
    name = f'{star}'
    target = table['Gaia source ID_1']==star
    if 'ZTF' in table['Light curves_1'][target].iloc[0]:
        inst = 'IRSA_ZTF'
    elif 'ATLAS' in table['Light curves_1'][target].iloc[0]:
        inst = 'ATLAS'
    elif 'ASAS-SN' in table['Light curves_1'][target].iloc[0]:
        inst = 'ASAS-SN'
    else:
        print(0)
        continue
    
    try:
        source = pd.read_csv(f'{inst}_lightcurves_std/{name}.csv')
    except:
        continue
    
    if inst == 'IRSA_ZTF':
        available_filters = list(set(source['filter']))
        if 'zg' in available_filters:
            filt = 'zg'
        else:
            filt = 'zr'
    
    if inst == 'ATLAS':
        filt = 'o'
        
    if inst == 'ASAS-SN':
        filt='g'
    
    tt = np.array(source['bjd'][source['filter']==filt])
    yy = np.array(source['mag'][source['filter']==filt])
    plots.fig_plot(tt, yy)
    
    # In order to perform our analysis we need to apply hybrid2d method
    # Input parametars are time data, magnitude data and parameteres for WWZ transformation, ntau, ngrid - grid size
    # As output main products are wwz matrix and autocorrelation matrix of wwz
    
    wwz_matrix, corr, extent = hybrid2d(tt, yy, 2000, 10000, minfq=20,maxfq=0.05)
    
    # Plotting wwz matrix heatmap
    # Note how er can easly spot detected period
    plots.plt_freq_heatmap(corr, extent)
      
    # Function for calculations of periods
    peaks0, hh0, r_periods0, up0, low0 = periods(name, corr, 10000, plot=True, minfq=20, maxfq=0.05)
    
    # Blue vertical lines represent the border of a half width of a peak
    # Green and red horizontal lines represent the calculated values of a quantiles
    # Purple line is a calculated value of half width of a peak
    
    # Write down obtained results
    for j in range(len(r_periods0)):
        print("Period: %6.3f, upper error : %5.2f  lower error : %5.2f"% (r_periods0[j], up0[j], low0[j]) )
        bulk_combine(f'{name}', [inst], 1/r_periods0[j], cycles=2, outliers='median', savepath=f'qhx/{name}_{r_periods0[j]}')

    
