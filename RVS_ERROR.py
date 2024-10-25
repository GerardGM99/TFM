# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:34:52 2024

@author: xGeeRe
"""

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.interpolate import interpn
from coord_utils import density_scatter

file = Table.read('data/rvs_bulk.csv', format='ascii.csv')
my_sources = Table.read('data/Radial_velocities.vot')

binsize = 0.2
mag = file['grvs_mag']
rv_err = file['radial_velocity_error']
nb_trans = file['rv_nb_transits']

mag_i = list(set(np.round(mag/binsize) * binsize))
mag_i.sort()
mag_i = np.array(mag_i)
mag_o = []
rv_err_o = []
e2_low = []
e2_upp = []
e1_low = []
e1_upp = []


for magi in mag_i:
    mask = np.abs(mag-magi) < binsize/2.
    mag_o.append(np.average(mag[mask]))
    rv_err_median = np.median(rv_err[mask])
    rv_err_o.append(rv_err_median)
    # e_low.append(rv_err_median-np.sqrt(pi/(2*len(mag[mask])))*(rv_err_median-np.percentile(rv_err[mask],15.85)))
    # e_upp.append(rv_err_median+np.sqrt(pi/(2*len(mag[mask])))*(np.percentile(rv_err[mask],84.15)-rv_err_median))
    e1_low.append(np.percentile(rv_err[mask],15.87))
    e1_upp.append(np.percentile(rv_err[mask],84.13))
    e2_low.append(np.percentile(rv_err[mask],2.28))
    e2_upp.append(np.percentile(rv_err[mask],97.72))
    
sources_list_silver=[6123873398383875456,5323384162646755712,2006088484204609408,2060841448854265216,
                     2061252975440642816,2166378312964576256,5311969857556479616,5524022735225482624,
                     6228685649971375616, 5880159842877908352,1870955515858422656,5965503866703572480,
                     2175699216614191360, 4076568861833452160,2206767944881247744]
   
plt.figure(figsize=(8,8/1.5))
plt.plot(mag_o, rv_err_o, color='darkviolet', label='Median')
# density_scatter(mag, rv_err, bins = 500, s = 2, alpha=0.1, cmap = "viridis")
plt.fill_between(mag_o, e2_low, e2_upp, alpha=0.2, color='r', label=r'2$\sigma$') #r'$2.28\%$-$97.72\%$'
plt.fill_between(mag_o, e1_low, e1_upp, alpha=0.2, color='b', label=r'1$\sigma$') #r'$15.87\%$-$84.13\%$'
# plt.scatter(my_sources['grvs_mag'], my_sources['radial_velocity_error'], label='My sources',
#             c='k', s=200, marker='*', edgecolor='gold')
plt.scatter(my_sources['grvs_mag'], my_sources['radial_velocity_error'], label='Our candidates',
            c=my_sources['rv_nb_transits'], s=130, marker='*', cmap='inferno', vmax=35, zorder=3)
cb = plt.colorbar(pad = 0.053, orientation='vertical')
cb.set_label('rv_nb_transits [#]', fontsize = 13, labelpad = 6, rotation=270, verticalalignment='bottom')

silver_x=[]
silver_y=[]
for ide in sources_list_silver:
    mask = my_sources['SOURCE_ID']==ide
    if len(mask[mask==True])==1:
        silver_x.append(my_sources['grvs_mag'][mask])
        silver_y.append( my_sources['radial_velocity_error'][mask])
plt.scatter(silver_x, silver_y, label='EBS/Ell', s=220, facecolors='none', edgecolors='greenyellow', linewidth=4)    

cb.ax.yaxis.set_ticks_position('left')
cb.ax.tick_params('both', labelsize=14)
plt.xlabel('grvs_mag [mag]', fontsize=15)
plt.ylabel('radial_velocity_errror [$km \\: s^{-1}$]', fontsize=15)
plt.tick_params('both', labelsize=15)
#plt.ylim(top=20)
plt.grid(zorder=0)
plt.legend(loc='upper left', fontsize=12)
# plt.show()
plt.savefig('cool_plots/Thesis_figures/rvs_error.png', bbox_inches = "tight", format = "png")
plt.show()
plt.close()

# plt.figure(figsize=(12,8))
# #plt.plot(nb_trans_o, rv_err_o, color='r')
# #plt.fill_between(mag_o, e_low, e_upp, alpha=0.2, color='r')
# density_scatter(nb_trans, rv_err, bins = 500, s = 2, alpha=0.1, cmap = "viridis", label = 'Hgap stars')
# plt.scatter(my_sources['rv_nb_transits'], my_sources['radial_velocity_error'], label='My sources',
#             c='k', s=200, marker='*', edgecolor='gold')
# plt.xlabel('rv_nb_transits', fontsize=16)
# plt.ylabel('radial_velocity_error [$km \\: s^{-1}$]', fontsize=16)
# plt.tick_params('both', labelsize=14)
# plt.ylim(bottom=0)
# plt.xlim(right=120, left=0)
# plt.grid()
# plt.legend(loc='upper right', fontsize=18)
# plt.show()
# plt.close()