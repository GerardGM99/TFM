# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:15:25 2024

@author: Gerard Garcia
"""


from Finker_script import use_finker
    
lc_directory='test_dir'
use_finker(lc_directory, freq_range=86400, freq_step=250000)

instru = ['ZTF']
for ins in instru:
    lc_directory=f'{ins}_lightcurves_std'
    use_finker(lc_directory, freq_range=100, freq_step=250000)

#%%
from lightcurve_utils import vel_period_mass
import numpy as np

m1 = np.linspace(0.5,8,25)
q = np.linspace(0.001,1,25)
period=[1/(5*51.293492)]
# m1 = [2.0825508, 2.7065687,2.0011842,0.8372387,2.152888,0.73,
#       2.4469435,0.9611464,0.832294,0.9001591,3.9091423,2.6136153,3.82779]
# q = np.linspace(0.001, 1, 25)
# period = [11.281873/24, 7.988562, 2.792011, 10.30832, 0.7934736, 0.95547, 
#           60.544959, 2.1391675, 72.3996845, 42.21887, 2.474583, 2.596063, 9.250328/24]

for  P, mass in zip(period,m1):
    m1 = np.repeat(mass, 25)
    vel_period_mass(m1, q, P)
    
#%%
from lightcurve_utils import lc_folding
import pandas as pd

table = pd.read_csv('ztf_lightcurves_std/2007318661608788096.csv')
#band=list(set(table['filter']))[0]
t= table['mjd'] #.loc[table['filter']==band]
y=table['mag']
yerr=table['magerr']

frequencies = [63.979816, 0.040427, 0.040427*2]
for best_freq in frequencies:
    lc_folding(t, y, yerr, best_freq)
    
#%%

import numpy as np
import pandas as pd
import astropy.units as u
from lightcurve_utils import lomb_scargle

table = pd.read_csv('ztf_lightcurves_std/460686648965862528.csv')
#band=list(set(table['filter']))[0]
t = np.array(table['mjd'].loc[table['filter']=='zg']) * u.day
y = np.array(table['mag'].loc[table['filter']=='zg']) * u.mag
yerr = np.array(table['magerr'].loc[table['filter']=='zg']) * u.mag

# frequency, power = LombScargle(t, y, dy=yerr).autopower(minimum_frequency=0.001/u.d,
#                                                         maximum_frequency=100/u.d)

# best_freq = frequency[np.argmax(power)]
# print(f'Best frequency: {best_freq}')

# plt.figure(figsize=(8,6))
# plt.plot(frequency, power)
# plt.show()

lomb_scargle(t, y, yerr, fmin=0.001,fmax=100, plot=True)