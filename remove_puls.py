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
from scipy.interpolate import interp1d


file = Table.read('data/others/ZTF19aavpxcc.csv', format='ascii.csv')

binsize=50

band='zr'

# Remove outliers
# sigma = remove_outliers(file['mag'][file['filter']==band])
sigma = file["mjd"]-58190.45>300
                
# Select time, magnitudes and magnitude errors
t_observed = np.array(file["mjd"][file['filter']==band])[sigma]
y_observed = np.array(file['mag'][file['filter']==band])[sigma]
y_observed_copy = np.copy(y_observed)
uncert = np.array(file["magerr"][file['filter']==band])[sigma]
uncert_copy = np.copy(uncert)


# Binnning
t_bin, y_bin, _, _ =  phot_utils.bin_lightcurve(t_observed, y_observed, yerr=uncert, 
                                                binsize=binsize, mode="average")

interp_func = interp1d(t_bin, y_bin, kind='linear')
fit_mask = (t_observed>58610)*(t_observed<60218)
new_y = interp_func(t_observed[fit_mask])

y_observed[fit_mask] = (y_observed[fit_mask] / new_y)
uncert[fit_mask] = uncert[fit_mask] / y_observed_copy[fit_mask]

# x_i = list(set(np.round(t_observed/binsize) * binsize))
# x_i.sort()
# x_i = np.array(x_i)
# average = np.mean(y_observed)
# i=0
# for t in t_bin:
#     mask = np.abs(t_observed-t_bin) < binsize/2
#     y_observed[mask] = (y_observed[mask] - y_bin[i]) #+ average
#     i+=1

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(9,5.5)
ax.grid(alpha=0.5)
ax.errorbar(t_observed[fit_mask] - 58190.45, y_observed[fit_mask], yerr=uncert[fit_mask], color='r', label=band, 
            fmt = "o", capsize=2, elinewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.4)
ax.set_xlabel("MJD - 58190.45 [days]", family = "serif", fontsize = 16)
ax.set_ylabel("Detrended Magnitude (no units)", family = "serif", fontsize = 16)
ax.tick_params(which='major', width=2, direction='out')
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, direction='out')
ax.tick_params(labelsize = 16)
ax.minorticks_on()
ax.invert_yaxis()
plt.show()
plt.close()

fig, ax2 = plt.subplots(constrained_layout=True)
fig.set_size_inches(9,5.5)
ax2.grid(alpha=0.5)
ax2.errorbar(t_observed - 58190.45, y_observed_copy, yerr=uncert_copy, color='r', label=band, 
            fmt = "o", capsize=2, elinewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.4)
ax2.plot(t_observed[fit_mask] - 58190.45, new_y, label = 'fit')
ax2.set_xlabel("MJD - 58190.45 [days]", family = "serif", fontsize = 16)
ax2.set_ylabel("Magnitude (mag)", family = "serif", fontsize = 16)
ax2.tick_params(which='major', width=2, direction='out')
ax2.tick_params(which='major', length=7)
ax2.tick_params(which='minor', length=4, direction='out')
ax2.tick_params(labelsize = 16)
ax2.minorticks_on()
ax2.invert_yaxis()
plt.legend()
plt.show()
plt.close()


#%%

from Finker_script import Finker_mag

t = t_observed[fit_mask]
y = y_observed[fit_mask]
yerr = uncert[fit_mask]

period_mask = t-58190.45 > 1500

freq = np.linspace(0.001,20,50000)
_,_,fig = Finker_mag(t[period_mask], y[period_mask], yerr[period_mask], freq, show_plot=True)

#%%
from lightcurve_utils import lc_folding


ran = np.arange(0, 5, 0.05)
# period = 38
for i in ran:
    p = 38 + i
    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(9,5.5)
    lc_folding('ZTF19aavpxcc', t, y, uncert=None, best_freq=1/p, ax=ax, t_start=t_observed[0])
    plt.show()
    plt.close()

#%%
from astropy.timeseries import LombScargle
import astropy.units as u

period_mask = t-58190.45 > 1500
t_l = t[period_mask] * u.day
y_l = y[period_mask] * u.mag
dy_l = yerr[period_mask] * u.mag

frequency, power = LombScargle(t_l, y_l, dy_l).autopower(minimum_frequency=0.001/u.day,
                                                   maximum_frequency=20/u.day)
plt.figure(figsize=(10,6))
plt.plot(frequency, power)
plt.xlabel('frequency [1/d]')
plt.ylabel('power')
plt.show()

print(frequency[np.argmax(power)])

# best_frequency = frequency[np.argmax(power)]
# ls = LombScargle(t, y, dy)
# model = ls.model(t, best_frequency)
