# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:15:25 2024

@author: Gerard Garcia
"""
#%%

from lightcurve_utils import standard_table

standard_table('MeerLICHT', 'data/ML_paul.csv', 'data/70_targets.csv', 'DR3_source_id')

#%%

from lightcurve_utils import plot_lightcurves

plot_lightcurves('IRSA_ZTF_lightcurves_std', outliers='eb')

#%%
from Finker_script import use_finker
    
lc_directory='IRSA_ZTF_lightcurves_std'
tab = use_finker(lc_directory, freq_range=100, freq_step=250000)

# instru = ['ZTF']
# for ins in instru:
#     lc_directory=f'{ins}_lightcurves_std'
#     use_finker(lc_directory, freq_range=100, freq_step=250000)

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
from lightcurve_utils import lc_folding, lc_combined, bulk_combine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# table = pd.read_csv('irsa_ztf_lightcurves_std/2060841448854265216.csv')
# band1=list(set(table['filter']))[0]
# t1= table['mjd'].loc[table['filter']==band1]
# y1=table['mag'].loc[table['filter']==band1]
# yerr1=table['magerr'].loc[table['filter']==band1]
# min_index = np.argmax(y1)
# t_start = t1.iloc[min_index]
# filt1 = table['inst'][0] + ', ' + band1

# # table = pd.read_csv('ztf_lightcurves_std/460686648965862528.csv')
# table = pd.read_csv('irsa_ztf_lightcurves_std/2060841448854265216.csv')
# band2=list(set(table['filter']))[1]
# t2= table['mjd'].loc[table['filter']==band2]
# y2=table['mag'].loc[table['filter']==band2]
# yerr2=table['magerr'].loc[table['filter']==band2]
# filt2 = table['inst'][0] + ', ' + band2

# # table = pd.read_csv('ztf_lightcurves_std/460686648965862528.csv')
# table = pd.read_csv('irsa_ztf_lightcurves_std/2060841448854265216.csv')
# band3=list(set(table['filter']))[2]
# t3= table['mjd'].loc[table['filter']==band3]
# y3=table['mag'].loc[table['filter']==band3]
# yerr3=table['magerr'].loc[table['filter']==band3]
# filt3 = table['inst'][0] + ', ' + band3

# lc_combined('Gaia DR3 2060841448854265216', [t1, t2, t3], [y1, y2, y3], [yerr1, yerr2, yerr3], [filt1, filt2, filt3], 
#             0.016517, t_start=t_start)

bulk_combine('2060841448854265216', ['ATLAS', 'IRSA_ZTF', 'ASAS_SN'], 1/60.545008)

# plt.tight_layout()
# plt.show()
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

#%%

from finder_chart import draw_image
import pandas as pd
import coord_utils

asciicords = pd.read_csv('data/70_targets.csv')
for ra, dec, name in zip(asciicords['ra'], asciicords['dec'], asciicords['DR3_source_id']):
    print(f'Starting with {name}')
    draw_image(ra, dec, name, 20/3600, directory='cool_plots/images/Nadia', ext='png',save=True,plot=False)
    coord_utils.draw_image(ra, dec, name, 160, directory='cool_plots/images/Color', ext='png',save=True,plot=False)
     
#%%

import spectra_utils as su

_,_,_,_ = su.model_spectral_lines([6553, 6573], {'Ha':6563}, [0.8], 5000, 7000, 10, cont=False, rv=[0])
#_,_,_,_ = su.model_spectral_lines([4000, 7000], {'Ha':6563}, [0.8], 25000, 7000, 20, cont=True, rv=[-20, 20])
#su.rv_crosscorr_err(n_boot, wl_range, n_points, lines, line_strenght, R, T, SNR)

#%%

import coord_utils as cu
from astropy.table import Table

random = Table.read('data/random_sources-result.vot', format="votable")
random = random.to_pandas()

mysources = Table.read('data/70_targets_coord.vot', format="votable")
mysources = mysources.to_pandas()

cu.sky_plot(random['ra'], random['dec'], projection=None, frame='icrs', density=True, label='bulk', s=10, cmap='inferno')
ax1 = plt.gca()
cu.sky_plot(mysources['ra'], mysources['dec'],projection=None, frame='icrs', c='w',marker='*',s=200, label='Test', ax=ax1)

plt.tight_layout()
plt.show()


#%%

import finder_chart as fc
from astropy.table import Table
import numpy as np
import pandas as pd

t_coord = Table.read('data/20_Calar.csv', format='ascii.csv')
separator = "|"
print ( "%s %s %s %s %s %s %s %s %s"%("# Object".ljust(20), separator, "alpha".ljust(11), separator, "delta".ljust(12), separator, "equinox".ljust(6), separator, "comments") )

name, ra, dec, com = [], [], [], []
eq = np.repeat('2000', len(t_coord))
for ide in t_coord['source_id']:
    mask = t_coord['source_id']==ide
    ra_h, dec_h = fc.deg2hour(t_coord['ra'][mask][0], t_coord['dec'][mask][0], sep=":")
    mag = round(t_coord['phot_g_mean_mag'][mask][0], 2)
    Texp = t_coord['Texp'][mask][0]
    comment = 'mag: G = '+str(mag)+'; Texp(s): '+ str(Texp)
    print ( "%s %s %s %s %s %s 2000.0  %s %s"%(str(ide).ljust(20), separator, ra_h, separator, dec_h, separator, separator, comment) )
    name.append(ide)
    ra.append(ra_h)
    dec.append(dec_h)
    com.append(comment)

dic = {'Object':name, 'alpha':ra, 'delta':dec, 'equinox':eq, 'comments':com}
calar_table = Table(dic)
calar_table.write('data/calar_targets_ipac.txt', format='ipac', overwrite=True)

#%%

from astropy.table import Table
import numpy as np
import pandas as pd

separator='|'
print ( "%s %s %s %s %s %s %s"%("# Name".ljust(12), separator, "ra".ljust(7), separator, "dec".ljust(7), separator, "mag") )
name=['AT 2024gfa', 'AT 2024gbc', 'AT 2024gfg', 'AT 2024gai', 'AT 2024fpb', 'AT 2024ggi', 'AT 2024dgr', 'AT 2024evp']
ra=['160.698', '218.631', '269.483', '266.020', '287.460', '169.592', '288.192', '236.548']
dec=['+27.662', '+58.310', '+85.550', '-13.359', '-17.939', '-32.838', '-19.269', '-10.057']
mag=['17.10', '18.18', '18.19', '17.90', '17.90', '18.90', '16.63', '16.26']

for i in range(len(name)):
    print ( "%s %s %s %s %s %s %s"%(name[i].ljust(12), separator, ra[i], separator, dec[i], separator, mag[i]) )
    
t = Table({'name':name, 'ra':ra, 'dec':dec, 'mag':mag})
t.write('Calar_transient_list.txt', format='ascii')

#%%

from finder_chart import get_finder

name=['AT2024gfg', 'AT2024fpb', 'AT2024ggi', 'AT2024dgr']
ra=[269.483, 287.460, 169.592, 288.192]
dec=[+85.550, -17.939, -32.838, -19.269]
mag_transient=[18.19, 17.90, 18.92, 16.83]

for i in range(len(name)):
    get_finder(ra[i], dec[i], name[i], 3/60, debug=True,
               starlist=None, print_starlist=True, telescope="Calar",
               directory="calar_finders", minmag=11, maxmag=13, mag=mag_transient[i],
               image_file=None)

#%%
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from lightcurve_utils import next_transits

thing = Table.read('IRSA_ZTF_lightcurves_std/2006088484204609408.csv', format='ascii.csv')

period = 2.596044
y = thing['mag']
time = Time(thing['mjd'], format='mjd')
# Set the eclipse time to the time when the maximum magnitude was observed
t_eclipse = time[np.argmax(y)]
# Also set the magnitude value at primary mid-eclipse
y_eclipse = y[np.argmax(y)]
# The median of the y-values gives an estimate of the magnitude out of eclipse
y_median = np.median(y)


period = period*u.d
mintime = Time(min(time))  # Starting time of the lightcurve
t_eclipse = Time(t_eclipse, format='mjd')  
n_eclipses = int((max(time)-min(time))/period)

# Calculated eclipses
primary_eclipses, secondary_eclipses = next_transits('', mintime, t_eclipse, period, 
                                                     n_eclipses=n_eclipses, verbose=False)

plt.figure()
plt.scatter(thing['mjd'], thing['mag'], s=5)
for x in primary_eclipses.value:
    plt.axvline(x, ls='-', alpha=0.5)
plt.xlim(58250, 58275)
plt.ylim(14.55, 14.1)
plt.show()

#%%

from astropy.table import Table
from lightcurve_utils import oc_diagram

# tab = Table.read('IRSA_ZTF_lightcurves_std/2060841448854265216.csv', format='ascii.csv')
# tab = Table.read('IRSA_ZTF_lightcurves_std/2006088484204609408.csv', format='ascii.csv')
tab = Table.read('IRSA_ZTF_lightcurves_std/2166378312964576256.csv', format='ascii.csv')
tab = tab[tab['filter']=='zr']

time = tab['mjd']
y = tab['mag']
period = 72.430412

# n, oc = oc_diagram(time, y, period)
n, oc = oc_diagram(time, y, period, t_eclipse=58357.31640625, y_eclipse=15)