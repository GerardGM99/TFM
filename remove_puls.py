# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:52:46 2024

@author: xGeeRe
"""
import os
from astropy.table import Table
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightcurve_utils import remove_outliers
import phot_utils


file = Table.read('ztf_lightcurves_std/4281886474885416064.csv', format='ascii.csv')

binsize=100

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(9,5.5)
ax.grid(alpha=0.5)

band='zg'

# Remove outliers
sigma = remove_outliers(file['mag'][file['filter']==band])
                
# Select time, magnitudes and magnitude errors
t_observed = np.array(file["mjd"][file['filter']==band])[sigma]
y_observed = np.array(file['mag'][file['filter']==band])[sigma]
uncert = np.array(file["magerr"][file['filter']==band])[sigma]

t_bin, y_bin, _, _ =  phot_utils.bin_lightcurve(t_observed, y_observed, yerr=uncert, 
                                                binsize=binsize, mode="average")

x_i = list(set(np.round(t_observed/binsize) * binsize))
x_i.sort()
x_i = np.array(x_i)
average = np.mean(y_observed)
i=0
for xi in x_i:
    mask = np.abs(t_observed-xi) < binsize/2
    y_observed[mask] = (y_observed[mask] - y_bin[i]) + average
    i+=1

ax.errorbar(t_observed - 58190.45, y_observed, yerr=uncert, color='g', label=band, 
            fmt = "o", capsize=2, elinewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.4)

band='zr'

# Remove outliers
sigma = remove_outliers(file['mag'][file['filter']==band])
                
# Select time, magnitudes and magnitude errors
t_observed = np.array(file["mjd"][file['filter']==band])[sigma]
y_observed = np.array(file['mag'][file['filter']==band])[sigma]
uncert = np.array(file["magerr"][file['filter']==band])[sigma]

t_bin, y_bin, _, _ =  phot_utils.bin_lightcurve(t_observed, y_observed, yerr=uncert,
                                                binsize=binsize, mode="average")

x_i = list(set(np.round(t_observed/binsize) * binsize))
x_i.sort()
x_i = np.array(x_i)
average = np.mean(y_observed)
i=0
for xi in x_i:
    mask = np.abs(t_observed-xi) < binsize/2
    y_observed[mask] = ((y_observed[mask] - y_bin[i])) + average
    i+=1

ax.errorbar(t_observed - 58190.45, y_observed, yerr=uncert, color='r', label=band, 
            fmt = "o", capsize=2, elinewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.4)


ax.invert_yaxis()
plt.show()

#%%

from Finker_script import Finker_mag

freq = np.linspace(0.001,20,50000)
_,_,fig = Finker_mag(t_observed, y_observed, uncert, freq, show_plot=True)

#%%
from lightcurve_utils import lc_folding

frequencies = [0.004242*5]
for best_freq in frequencies:
    lc_folding(t_observed, y_observed, uncert=None, best_freq=best_freq)
    
#%%
from astropy.timeseries import LombScargle
import astropy.units as u

t = t_observed * u.day
y = y_observed * u.mag
dy = uncert * u.mag

frequency, power = LombScargle(t, y, dy).autopower(minimum_frequency=0.001/u.day,
                                                   maximum_frequency=1000/u.day)
plt.figure()
plt.plot(frequency, power)
plt.show()

# best_frequency = frequency[np.argmax(power)]
# ls = LombScargle(t, y, dy)
# model = ls.model(t, best_frequency)
