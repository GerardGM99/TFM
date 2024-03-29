# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:15:25 2024

@author: Gerard Garcia
"""
#%%

from lightcurve_utils import standard_table

standard_table('ZTF', 'ZTF_lightcurves', 'data/70_targets.csv', 'DR3_source_id')

#%%

from lightcurve_utils import plot_lightcurves

plot_lightcurves('BLACKGEM_lightcurves_std')

#%%
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
import matplotlib.pyplot as plt

table = pd.read_csv('ztf_lightcurves_std/2006088484204609408.csv')
band=list(set(table['filter']))[0]
t1= table['mjd'].loc[table['filter']==band]
y1=table['mag'].loc[table['filter']==band]
yerr1=table['magerr'].loc[table['filter']==band]

table = pd.read_csv('ztf_lightcurves_std/460686648965862528.csv')
band=list(set(table['filter']))[1]
t2= table['mjd'].loc[table['filter']==band]
y2=table['mag'].loc[table['filter']==band]
yerr2=table['magerr'].loc[table['filter']==band]

# frequencies = [1/52.7027]
# for best_freq in frequencies:
#     lc_folding(t, y, yerr, best_freq)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

lc_folding('Gaia DR3 2006088484204609408', t1, y1, yerr1, 0.385202, axs[0], cycles=2)
lc_folding('Gaia DR3 460686648965862528', t2, y2, yerr2, 3.190962, axs[1], cycles=2)


plt.tight_layout()
plt.show()
#%%

import numpy as np
import pandas as pd
import astropy.units as u
from lightcurve_utils import lomb_scargle

table = pd.read_csv('ztf_lightcurves_std/2061252975440642816.csv')
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

lomb_scargle(t, y, yerr, fmin=100, fmax=300, plot=True)

#%%

from gaiaxpy import convert, calibrate
from gaiaxpy import plot_spectra
import pandas as pd
import numpy as np
from spectra_utils import Gaia_XP

sources_list=[5962956195185292288,4519475166529738112,4515124540765147776,4321276689423536384,5593826360487373696,
              2027563492489195520,2060841448854265216,2061252975440642816,2166378312964576256,2006912396372680960,
              2007318661608788096,4299904519833646080,3444168325163139840,2002117151282819840,6228685649971375616,
              508419369310190976,2013187240507011456,2054338249889402880,2074693061975359232,2164630463117114496,
              175699216614191360,428103652673049728,527155253604491392,4263591911398361472,3355776901779440384,
              5350869719969619840,4054010697162430592,4094491141885400576,461193695624775424,
              512721444765993472,3369399099232812160,2931553674771048704,4096527235637366912,
              5599309216965305728,6123873398383875456,5882737819707242240,2934216142176785920,
              2200433413577635840,4076568861833452160,5866474526572151936,473575777103322496,
              2030965725082217088,2163542397602876800,526939882463713152,2173852964799716480,
              5524022735225482624,2169083008475385856,2206767944881247744,187219239343050880,
              5617186348318629248,6053890788968694656,5880159842877908352,5965503866703572480,
              1870955515858422656,2006088484204609408,5311969857556479616,5338183383022960512,
              5866345647515558400,4281886474885416064,4272588356022299520,4271992661242707200,
              4260141158544875008,2083649030845658624,5868425648663616768,460686648965862528,
              5323384162646755712,5328449200388495616,6052463412400724992,5966574133883852416,5326288831829996416]

# sources_list=[5962956195185292288,4519475166529738112]

Gaia_XP(sources_list, out_path='SPECTRA')

# calibrated_spectra, sampling = calibrate(sources_list)

# for i in range(len(sources_list)):
#     source = calibrated_spectra.iloc[[i]]
#     plt.figure(figsize=(12,6))
#     plt.subplot()
#     plt.errorbar(sampling, np.array(source['flux'])[0], yerr=np.array(source['flux_error'])[0], fmt=".-", color="k", label = "DR3")


#%%

from astropy.table import Table
import matplotlib.pyplot as plt

table = Table.read('ztf_lightcurves_std/2061252975440642816.csv', format='ascii.csv')
table = table[table['filter'] == 'zr']

t_obs = table['mjd'][(table['mjd']- 58190.45<246.8)&(table['mjd']- 58190.45>246.6)]
y_obs = table['mag'][(table['mjd']- 58190.45<246.8)&(table['mjd']- 58190.45>246.6)]
yerr = table['magerr'][(table['mjd']- 58190.45<246.8)&(table['mjd']- 58190.45>246.6)]

fig, ax = plt.subplots(figsize=(15, 8))
ax.errorbar(t_obs - 58190.45, y_obs, yerr=yerr, color = 'r',
             fmt = "o", capsize=2, elinewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.4)
ax.invert_yaxis()
ax.set_ylabel("Magnitude", family = "serif", fontsize = 16)
ax.set_xlabel("MJD - 58190.45 [days]", family = "serif", fontsize = 16)
plt.show()

#%%

#from lightcurve_utils import plot_ind_lightcurves
from lightcurve_utils import plot_lightcurves

plot_lightcurves('BLACKGEM_lightcurves_std', rang='single', plot=True, savefig=False)