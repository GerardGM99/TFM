# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:55:13 2024

@author: xGeeRe
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interpn
from astropy.table import Table

#Function to plot sources as a scatter plot with density
def density_scatter( x , y, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    #if ax is None :
        #fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    out = plt.scatter( x, y, c=z, **kwargs )
    #plt.colorbar(label = "Sources per bin", location="left")
        
    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm))
    #cbar.ax.set_ylabel('Density')

    return out

random = Table.read('data/10mili.vot', format="votable")
random = random.to_pandas()

mysources = Table.read('data/70_targets_coord.vot', format="votable")
mysources = mysources.to_pandas()

#%%
#SKYMAP
#plt.style.use('dark_background')
plt.figure(figsize=(12, 8))
plt.subplot(projection="aitoff")

coords = SkyCoord(frame = "galactic", l=random['l'], b=random['b'], unit='degree')
l = -coords.l.wrap_at(180 * u.deg).radian
b = coords.b.radian
#plt.scatter(l, b, c="b", s=2, alpha=0.3, label="Random Gaia sources")
density_scatter(l, b, bins = 500, s = 2, alpha=0.5, cmap = "plasma")
#plt.colorbar(orientation="horizontal", pad=0.2)

coords = SkyCoord(l=mysources['l'][mysources['phot_g_mean_mag']<13], 
                  b=mysources['b'][mysources['phot_g_mean_mag']<13], 
                  unit='degree', frame = "galactic")
l = -coords.l.wrap_at(180 * u.deg).radian
b = coords.b.radian
plt.scatter(l, b, c="w", marker="*", edgecolors='k', s=220, label="Mag<13")

coords = SkyCoord(l=mysources['l'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
                  b=mysources['b'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
                  unit='degree', frame = "galactic")
l = -coords.l.wrap_at(180 * u.deg).radian
b = coords.b.radian
plt.scatter(l, b, c="w", marker="*", edgecolors='k', s=120, label="13<Mag<15")


coords = SkyCoord(l=mysources['l'][mysources['phot_g_mean_mag']>15], 
                  b=mysources['b'][mysources['phot_g_mean_mag']>15], 
                  unit='degree', frame = "galactic")
l = -coords.l.wrap_at(180 * u.deg).radian
b = coords.b.radian
plt.scatter(l, b, c="w", marker="*", edgecolors='k', s=40, label="Mag>15")



# Convert the longitude values in right ascension hours
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, \
                              30, 60, 90, 120, 150]),
            labels=['10h', '8h', '6h', '4h', '2h', '0h', \
                    '22h', '20h', '18h', '16h', '14h'])

# Plot the labels and the title
plt.title("Skymap in galactic coordinates" , x = 0.5, y = 1.1, fontsize=19)
plt.xlabel('l')
plt.ylabel('b')

# Grid and legend
plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1,1.11))
plt.grid(True)

plt.show()
plt.close()

#%%
plt.figure(figsize=(12,8))

#Labels and title
plt.title("Skymap in ICRS", fontsize=19)
plt.xlabel("ra (deg)", fontsize = 19)
plt.ylabel("dec (deg)", fontsize = 19)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)


density_scatter(random['ra'], random['dec'], bins = 500, s = 2, alpha=0.5, cmap = "plasma")
cb = plt.colorbar(pad = 0.037)
cb.set_label("Sources per bin", fontsize = 19, labelpad = -75)


plt.scatter(mysources['ra'][mysources['phot_g_mean_mag']<13], 
            mysources['dec'][mysources['phot_g_mean_mag']<13], 
            c="w", marker="*", edgecolors='k', s=220, label="Mag<13")
plt.scatter(mysources['ra'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
            mysources['dec'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
            c="w", marker="*", edgecolors='k', s=120, label="13<Mag<15")
plt.scatter(mysources['ra'][mysources['phot_g_mean_mag']>15], 
            mysources['dec'][mysources['phot_g_mean_mag']>15], 
            c="w", marker="*", edgecolors='k', s=40, label="Mag>15")

plt.legend(loc='upper center', fontsize=12, shadow=True)
plt.show()
plt.close()

#%%
#SKYMAP
plt.figure(figsize=(12, 8))
plt.subplot(projection="aitoff")

coords = SkyCoord(frame = "icrs", ra=random['ra'], dec=random['dec'], unit='degree')
ra_rad = coords.ra.wrap_at(180 * u.deg).radian
dec_rad = coords.dec.radian
density_scatter(ra_rad, dec_rad, bins = 500, s = 2, alpha=0.5, cmap = "plasma")
#plt.colorbar(location='bottom', pad=0.1)

coords = SkyCoord(ra=mysources['ra'][mysources['phot_g_mean_mag']<13], 
                  dec=mysources['dec'][mysources['phot_g_mean_mag']<13], 
                  unit='degree', frame = "icrs")
ra_rad = coords.ra.wrap_at(180 * u.deg).radian
dec_rad = coords.dec.radian
plt.scatter(ra_rad, dec_rad, c="w", marker="*", edgecolors='k', s=220, label="Mag<13")

coords = SkyCoord(ra=mysources['ra'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
                  dec=mysources['dec'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
                  unit='degree', frame = "icrs")
ra_rad = coords.ra.wrap_at(180 * u.deg).radian
dec_rad = coords.dec.radian
plt.scatter(ra_rad, dec_rad, c="w", marker="*", edgecolors='k', s=120, label="13<Mag<15")

coords = SkyCoord(ra=mysources['ra'][mysources['phot_g_mean_mag']>15], 
                  dec=mysources['dec'][mysources['phot_g_mean_mag']>15], 
                  unit='degree', frame = "icrs")
ra_rad = coords.ra.wrap_at(180 * u.deg).radian
dec_rad = coords.dec.radian
plt.scatter(ra_rad, dec_rad, c="w", marker="*", edgecolors='k', s=40, label="Mag>15")


# Convert the longitude values in right ascension hours
# plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, \
#                               30, 60, 90, 120, 150]),
#             labels=['10h', '8h', '6h', '4h', '2h', '0h', \
#                     '22h', '20h', '18h', '16h', '14h'])

# Plot the labels and the title
plt.title("Skymap in ICRS" , x = 0.5, y = 1.1, fontsize=19)
plt.xlabel('ra')
plt.ylabel('dec')

# Create a grid
plt.grid(True)
plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1,1.11))

plt.show()
plt.close()