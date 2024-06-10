# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:15:48 2024

@author: xGeeRe
"""
import numpy as np
from scipy.constants import c
from astropy.io import fits
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams

rcParams['xtick.minor.size'] = 6
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['xtick.minor.size'] = 4
rcParams['ytick.minor.size'] = 4

rcParams['font.size']= 18
rcParams['font.family']= 'sans-serif'
rcParams['xtick.major.width']= 2.
rcParams['ytick.major.width']= 2.

def abell_rad(z):
    return 1.72 *(1+z)**2 / (z*(1+z/2))

def R_vir(V_pec):
    H0 = 70 #km/s/Mpc
    return np.sqrt(3) * V_pec / (10*H0)
    
def M_vir(V_pec, R_vir):
    G = 4.301e-9 #km^2 Mpc Msun^-1 s^-2
    return 3*V_pec**2 * R_vir / G

def sigma_i(zi, zclu):
    return (c*(zi - zclu)/(1+zclu))/1000

def sigma_clu(Ngal, vel):
    return np.sum(vel**2)/Ngal

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

G = 4.301e-9 #Mpc Msun^-1 (km/s)^2
H0 = 70 #km/s/Mpc
#clusters have mean mass of about 10^14 Msun, so we can take 6 for the concentration parameter
C_v = 6
rho_clus = 1.39e11 #Msun/Mpc^3
rho_s = (C_v**3)/3 * 1/(np.log(1+C_v) - C_v/(1+C_v)) * 100 * rho_clus

def Phi(r, r_2):
    return - (4 * np.pi * G * rho_s * r_2**3) / r * np.log(1 + r/r_2)

def beta(r, r_2):
    return 0.5 * r / (r + r_2)

def v_esc(r, r_2):
    return np.sqrt(- 2 * Phi(r, r_2) * (1 - beta(r, r_2))/(3 - 2*beta(r, r_2)))

table = fits.open('sdss_gals', memmap=True)

data = table[1].data

df = pd.DataFrame(data)

dictionary = {'ID':df['dr8objid'],
              'ra': df['RA_nsatlas'], 'dec':df['DEC_nsatlas'],
              'redshift': df['z_hel'], 
              'u':df['ELPETRO_ABSMAG_u'],
              'g':df['ELPETRO_ABSMAG_g'],
              'r':df['ELPETRO_ABSMAG_r'],
              'i':df['ELPETRO_ABSMAG_i'],
              'z':df['ELPETRO_ABSMAG_z'],
              'type':df['TType']}

tab = Table(dictionary)
tab = tab[(tab['ID']!=-9999)]

cluster_coords = SkyCoord(230.757500, 8.639444, unit='deg')
galaxy_coords = SkyCoord(tab['ra'], tab['dec'], unit='deg')
sep_constraint = galaxy_coords.separation(cluster_coords) < 2*abell_rad(0.0340) * u.arcmin
a2063 = tab[sep_constraint]


# mu, sigma = norm.fit(a2063['redshift'][(a2063['redshift']<0.041)&(a2063['redshift']>0.028)])
# print(fr'readshift mean of the Gaussian fit: {mu}\pm {sigma}')
# print('Simbad redshift: ', 0.034)
# mu2, sigma2 = norm.fit(a2063['redshift'][(a2063['redshift']>0.041)&(a2063['redshift']<0.05)])
# x = np.linspace(0.018, 0.052, 1000)
# plt.figure(figsize=(8,6))
# plt.hist(a2063['redshift'][(a2063['redshift']<0.05)], bins=50, color='r', label='original data', alpha=0.7)
# mask_redshift = abs(a2063['redshift']-mu) < 3*sigma
# plt.axvspan(0.01, mu-3*sigma, alpha=0.3, color='k')
# plt.axvspan(mu+3*sigma, 0.055, alpha=0.3, color='k')
# # plt.hist(a2063['redshift'][mask_redshift], bins=23, color='b', histtype=u'step', lw=3, label='redshift cut')
# plt.plot(x, gaussian(x, 30, mu, sigma), color='k', label='Gaussian fit')
# # plt.plot(x, gaussian(x, 24, mu2, sigma2), color='k')
# plt.axvline(mu+3*sigma, ls='dashed', color='k', label=r'$\pm 3\sigma$')
# plt.axvline(mu-3*sigma, ls='dashed', color='k')
# plt.xlabel('Redshift', fontsize=15)
# plt.ylabel('Number of galaxies', fontsize=15)
# plt.tick_params(axis='both', labelsize=14)
# plt.xlim(left=0.018, right=0.052)
# plt.legend(fontsize=14, loc='upper left')
# plt.tight_layout()
# plt.show()

# mask_redshift = abs(a2063['redshift']-mu) < 3*sigma


a2063.add_column(sigma_i(a2063['redshift'],0.034), name='v_pec')
cut_a2063 = a2063
# cut_a2063 = a2063[mask_redshift]


#%%

mu, sigma = norm.fit(cut_a2063['v_pec'][abs(cut_a2063['v_pec'])<sigma_i(0.040,0.034)])
mu2, sigma2 = norm.fit(a2063['v_pec'][(a2063['v_pec']>sigma_i(0.040,0.034))&(a2063['v_pec']<4000)])
x = np.linspace(-3000, 3000, 3000)
x2 = np.linspace(0, 5000, 3000)
plt.figure(figsize=(8,6))
# plt.hist(a2063['v_pec'][abs(a2063['v_pec'])<5000], bins=80, color='b')
mask_vel = abs(cut_a2063['v_pec']-mu) < 3*sigma
# plt.hist(cut_a2063['v_pec'][mask_vel], bins=30, color='r', histtype=u'step', lw=3, label='velocity cut')
plt.hist(cut_a2063['v_pec'][cut_a2063['v_pec']<5000], bins=60, color='b', label='original data', alpha=0.7)
plt.axvline(mu+3*sigma, ls='dashed', color='k', label=r'$\pm 3\sigma$')
plt.axvline(mu-3*sigma, ls='dashed', color='k')
plt.axvspan(-6000, mu-3*sigma, alpha=0.3, color='k')
plt.axvspan(mu+3*sigma, 5000, alpha=0.3, color='k')
plt.plot(x, gaussian(x, 27, mu, sigma), color='k', label='Gaussian fit')
plt.plot(x2, gaussian(x2, 19, mu2, sigma2), color='r', label='MKW3s')
# plt.plot(x, gaussian(x, 30, mu2, sigma2), color='k')
plt.xlabel('Peculiar velocities [km/s]', fontsize=15)
plt.ylabel('Number of galaxies', fontsize=15)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(left=-4500, right=5000)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

our_cluster = cut_a2063[mask_vel]

mu1, sigma1 = norm.fit(our_cluster['redshift'])
print(fr'readshift mean of the Gaussian fit: {mu}\pm {sigma}')
print('Simbad redshift: ', 0.034)
x = np.linspace(0.018, 0.052, 1000)
plt.figure(figsize=(8,6))
plt.hist(our_cluster['redshift'], bins=25, color='r', alpha=0.7)
plt.plot(x, gaussian(x, 25, mu1, sigma1), color='k', label='Gaussian fit')
plt.xlabel('Redshift', fontsize=15)
plt.ylabel('Number of galaxies', fontsize=15)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(left=0.025, right=0.045)
plt.legend(fontsize=14, loc='upper left')
plt.tight_layout()
plt.show()

our_cluster['ra_clu'] = 230.757500
our_cluster['dec_clu'] = 8.639444
our_cluster['dist_clu'] = 3e5 * 0.0340 / H0
our_cluster['R_proj'] = our_cluster['dist_clu'] * np.pi/180 * np.sqrt((our_cluster['ra_clu'] - our_cluster['ra'])**2 * np.cos(our_cluster['dec_clu'])**2
                                                                      + (our_cluster['dec_clu'] - our_cluster['dec'])**2)

our_cluster['sig_clu'] = np.sqrt(sigma_clu(len(our_cluster), our_cluster['v_pec']))
our_cluster['Rvir'] = R_vir(np.mean(our_cluster['sig_clu']))
our_cluster['v_esc'] = v_esc(our_cluster['R_proj'], our_cluster['Rvir']/6)
print(len(our_cluster))

our_cluster = our_cluster[abs(our_cluster['v_pec']) <= our_cluster['v_esc']]
our_cluster = our_cluster[our_cluster['R_proj']/our_cluster['Rvir'] <= 3]
print(len(our_cluster))

#%%

# Msol = 1.9891e30
print(r'$cz_{mean}$ = ', c*np.mean(our_cluster['redshift'])/1000, ' km/s')
sigmaclu =  np.sqrt(sigma_clu(len(our_cluster), our_cluster['v_pec']))
print(r'$\sigma_{clu}$ = ', sigmaclu, ' km/s')
Rvir = R_vir(sigmaclu)
print(r'$R_{vir}$ = ', R_vir(sigmaclu), ' Mpc')
print(r'$M_{vir}$ = ', M_vir(sigmaclu, Rvir), ' Msun')

#%%

def divider(x):
    return 2.06-0.244*np.tanh((x+20.07)/1.09)

our_cluster.sort('r')
y_div=divider(our_cluster['r'])

plt.figure(figsize=(8,6))
mask_red = (our_cluster['u']-our_cluster['r']) > y_div
mask_blue = (our_cluster['u']-our_cluster['r']) < y_div
plt.scatter(our_cluster['r'][mask_blue], our_cluster['u'][mask_blue]-our_cluster['r'][mask_blue], 
            s=15, color='b')
plt.scatter(our_cluster['r'][mask_red], our_cluster['u'][mask_red]-our_cluster['r'][mask_red], 
            s=15, color='r')
plt.plot(our_cluster['r'], y_div, color='green', ls='dashed')
plt.xlabel(r'Absolute magnitude, $M_r$', fontsize=15)
plt.ylabel('(u-r) color', fontsize=15)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.show()

#%%

from coord_utils import sky_plot

# sky_plot(our_cluster['ra'][(our_cluster['ra']<231)&(our_cluster['ra']>230.5)&(our_cluster['dec']<9)&(our_cluster['dec']>8.4)], 
#          our_cluster['dec'][(our_cluster['ra']<231)&(our_cluster['ra']>230.5)&(our_cluster['dec']<9)&(our_cluster['dec']>8.4)], 
#          projection=None, frame='icrs', color='k', label='galaxies', s=100)

sky_plot(tab['ra'], tab['dec'], xlimits=[225, 235.5], ylimits = [5.5, 11.5],
         projection=None, frame='icrs', color='k', label='galaxies', s=10)
ax = plt.gca()
sky_plot(our_cluster['ra'], our_cluster['dec'], 
         projection=None, frame='icrs', color='b', label='our cluster', s=20, marker='x', ax=ax)

sky_plot(230.757500, 8.639444, projection=None, frame='icrs', color='r', marker='X', label='cluster center', s=100, ax=ax)


#%%

theta = (np.pi/180)*our_cluster['ra']
r = our_cluster['dec']

theta2 = (np.pi/180)*a2063['ra']
r2 = a2063['dec']

theta3 = (np.pi/180)*tab['ra']
r3 = tab['dec']

colors = 'red'

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, polar=True)
# ax.scatter(theta2, r2, c='b', alpha=0.75, s=10)
ax.scatter(theta3, r3, c='k', alpha=0.75, s=5)
c = ax.scatter(theta, r, c='r', marker='x', s=5)


# ax.set_thetamin(228.5)
# ax.set_thetamax(233)

# ax.set_ylim(6.5, 10.5)
# ax.set_rorigin(-30)

ax.set_thetamin(225)
ax.set_thetamax(235.5)

ax.set_ylim(5.5, 11.5)
ax.set_rorigin(-15)

# ax.set_xlabel('ra')
# ax.set_ylabel('dec')
plt.show()

#%%

from phot_utils import density_scatter
import matplotlib.transforms as transforms



theta = (np.pi/180)*our_cluster['ra']
r = our_cluster['redshift']

theta2 = (np.pi/180)*a2063['ra']
r2 = a2063['redshift']

mask = (tab['dec'] > 6.952667095933982)*(tab['dec'] < 10.012771202261671)
theta3 = (np.pi/180)*tab['ra'][mask]
r3 = tab['redshift'][mask]


fig = plt.figure(figsize=(8,8), facecolor='darkorange')
ax = fig.add_subplot(111, polar=True)
ax.scatter(theta3, r3, c='k', alpha=0.75, s=3)
# density_scatter(theta3, r3, alpha=0.75, s=1, ax=ax)
# ax.scatter(theta2, r2, c='b', alpha=0.75, s=10)
# c = ax.scatter(theta, r, c='r', marker='x', s=5)


# ax.set_thetamin(228.5)
# ax.set_thetamax(233)

# ax.set_ylim(6.5, 10.5)
# ax.set_rorigin(-30)

ax.set_thetamin(224)
ax.set_thetamax(236.5)

ax.set_ylim(0.02, 0.065)
ax.set_rorigin(-0.07)

trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
ax.text(0.13, 0.74, 'redshift', weight='bold', transform = trans, rotation = 44)
ax.text(0.06, 0.03, 'RA (deg)', weight='bold', transform = trans, rotation = -43)
ax.tick_params(axis='both', labelsize=16)

# ax.set_xlabel('ra')
# ax.set_ylabel('z')
plt.show()

#%%
#red vs blue
def c_div(Mr):
    return 2.06 - 0.244 * np.tanh((Mr + 20.07)/1.09)

our_cluster.sort('r')


our_cluster['u_r'] = our_cluster['u'] - our_cluster['r']  

red_gals = our_cluster[our_cluster['u_r'] > c_div(our_cluster['r'])]


blue_gals = our_cluster[our_cluster['u_r'] < c_div(our_cluster['r'])]


plt.figure(figsize=(9,6))
plt.title('Red vs. blue galaxies')
plt.scatter(blue_gals['r'] + 20.07 - 5*np.log(H0), blue_gals['u_r'], marker='o', c='b', s=20)
plt.scatter(red_gals['r'] + 20.07 - 5*np.log(H0), red_gals['u_r'], marker='o', c='r', s=20)
plt.plot(our_cluster['r'] + 20.07 - 5*np.log(H0), c_div(our_cluster['r']), c='black', linestyle='dashed')
plt.xlabel('M_r + 20.07 - 5*log(H_0)')
plt.ylabel('M_u - M_r')
plt.savefig('red_vs_blue', bbox_inches='tight')

#%%

def region_A(X, Y):
    return ((4*X + Y > 4) & (Y > 1.5) & (Y < 2.5) | \
           (4*X + Y > 5) & (Y > 0.5) & (Y < 1.5) | \
           (4*X + Y > 6) & (Y < 0.5) & (Y >= 0)) & \
           (4*X + 7*Y < 19)

def region_B(X, Y):
    return (2 < 4*X + Y) & (4*X + Y < 4) & (Y > 1.5) & (Y < 2.5)

def region_C(X, Y):
    return (2 < 4*X + Y) & (4*X + Y < 5) & (Y > 0.5) & (Y < 1.5)

def region_D(X, Y):
    return (2 < 4*X + Y) & (4*X + Y < 6) & (Y < 0.5)

def region_E(X, Y):
    return (4*X + Y < 2)

def region_A_neg(X, Y):
    return ((-4*X + Y < -4) & (Y < -1.5) & (Y > -2.5) | \
           (-4*X + Y < -5) & (Y < -0.5) & (Y > -1.5) | \
           (-4*X + Y < -6) & (Y > -0.5) & (Y <= 0)) & \
           (-4*X + 7*Y > -19)

def region_B_neg(X, Y):
    return (-2 > -4*X + Y) & (-4*X + Y > -4) & (Y < -1.5) & (Y > -2.5)

def region_C_neg(X, Y):
    return (-2 > -4*X + Y) & (-4*X + Y > -5) & (Y < -0.5) & (Y > -1.5)

def region_D_neg(X, Y):
    return (-2 > -4*X + Y) & (-4*X + Y > -6) & (Y > -0.5)

def region_E_neg(X, Y):
    return -4*X + Y > -2

plt.figure(figsize=(8,8), facecolor='darkorange')

X_pos, Y_pos = np.meshgrid(np.linspace(0, 3, 1000), np.linspace(0, 2.5, 1000))

A_pos = np.array(region_A(X_pos, Y_pos))
B_pos = np.array(region_B(X_pos, Y_pos))
C_pos = np.array(region_C(X_pos, Y_pos))
D_pos = np.array(region_D(X_pos, Y_pos))
E_pos = np.array(region_E(X_pos, Y_pos))

A_pos = (A_pos*1).astype(float)
A_pos[A_pos==0] = np.nan
B_pos = (B_pos*1).astype(float)
B_pos[B_pos==0] = np.nan
C_pos = (C_pos*1).astype(float)
C_pos[C_pos==0] = np.nan
D_pos = (D_pos*1).astype(float)
D_pos[D_pos==0] = np.nan
E_pos = (E_pos*1).astype(float)
E_pos[E_pos==0] = np.nan

plt.annotate('A',(2,0.7), fontsize=25, weight='bold', alpha=0.5)
plt.annotate('B',(0.21,1.89), fontsize=25, weight='bold', alpha=0.5)
plt.annotate('C',(0.6,0.85), fontsize=25, weight='bold', alpha=0.5)
plt.annotate('D',(0.9,-0.05), fontsize=25, weight='bold', alpha=0.5)
plt.annotate('E',(0.15,0.4), fontsize=25, weight='bold', alpha=0.5)

plt.contourf(X_pos, Y_pos, A_pos, alpha=0.5, cmap='Blues', extend='both')
plt.contourf(X_pos, Y_pos, B_pos, alpha=0.5, cmap='Greens', extend='both')
plt.contourf(X_pos, Y_pos, C_pos, alpha=0.5, cmap='copper', extend='both')
plt.contourf(X_pos, Y_pos, D_pos, alpha=0.5, cmap='Oranges', extend='both')
plt.contourf(X_pos, Y_pos, E_pos, alpha=0.5, cmap='Reds', extend='both')

A_neg = np.array(region_A_neg(X_pos, -Y_pos))
B_neg = np.array(region_B_neg(X_pos, -Y_pos))
C_neg = np.array(region_C_neg(X_pos, -Y_pos))
D_neg = np.array(region_D_neg(X_pos, -Y_pos))
E_neg = np.array(region_E_neg(X_pos, -Y_pos))

A_neg = (A_neg*1).astype(float)
A_neg[A_neg==0] = np.nan
B_neg = (B_neg*1).astype(float)
B_neg[B_neg==0] = np.nan
C_neg = (C_neg*1).astype(float)
C_neg[C_neg==0] = np.nan
D_neg = (D_neg*1).astype(float)
D_neg[D_neg==0] = np.nan
E_neg = (E_neg*1).astype(float)
E_neg[E_neg==0] = np.nan

A_galax = region_A(our_cluster['R_proj']/our_cluster['Rvir'], abs(our_cluster['v_pec'])/abs(our_cluster['sig_clu']))
B_galax = region_B(our_cluster['R_proj']/our_cluster['Rvir'], abs(our_cluster['v_pec'])/abs(our_cluster['sig_clu']))
C_galax = region_C(our_cluster['R_proj']/our_cluster['Rvir'], abs(our_cluster['v_pec'])/abs(our_cluster['sig_clu']))
D_galax = region_D(our_cluster['R_proj']/our_cluster['Rvir'], abs(our_cluster['v_pec'])/abs(our_cluster['sig_clu']))
E_galax = region_E(our_cluster['R_proj']/our_cluster['Rvir'], abs(our_cluster['v_pec'])/abs(our_cluster['sig_clu']))

print('A: ', len(A_galax[A_galax==True]))
print('B: ', len(B_galax[B_galax==True]))
print('C: ', len(C_galax[C_galax==True]))
print('D: ', len(D_galax[D_galax==True]))
print('E: ', len(E_galax[E_galax==True]))

plt.contourf(X_pos, -Y_pos, A_neg, alpha=0.5, cmap='Blues', extend='both')
plt.contourf(X_pos, -Y_pos, B_neg, alpha=0.5, cmap='Greens', extend='both')
plt.contourf(X_pos, -Y_pos, C_neg, alpha=0.5, cmap='copper', extend='both')
plt.contourf(X_pos, -Y_pos, D_neg, alpha=0.5, cmap='Oranges', extend='both')
plt.contourf(X_pos, -Y_pos, E_neg, alpha=0.5, cmap='Reds', extend='both')

plt.scatter(blue_gals['R_proj']/blue_gals['Rvir'], blue_gals['v_pec']/blue_gals['sig_clu'], c='b', s=20)
plt.scatter(red_gals['R_proj']/red_gals['Rvir'], red_gals['v_pec']/red_gals['sig_clu'], c='r', s=20)

x = np.linspace(0,3*our_cluster['Rvir'][0],197)/our_cluster['Rvir'][0]

plt.plot(x, v_esc(x, our_cluster['Rvir']/6)/our_cluster['sig_clu'], color='black', label='<v_esc>')
plt.plot(x, -v_esc(x, our_cluster['Rvir']/6)/our_cluster['sig_clu'], color='black')

plt.tick_params(axis='both')
plt.xlabel(r'$R_{proj} \: / \: R_{vir}$')
plt.ylabel(r'$V_{LOS} \: / \: \sigma_{LOS}$')
plt.ylim(-3,3)
plt.savefig('infall.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False, facecolor='darkorange')
plt.subplots_adjust(wspace=0.2, hspace=0.1)

axs[0].hist(red_gals['v_pec'], color='r', alpha=0.7, histtype='step', lw=2, hatch='/', zorder=0)
axs[0].set_xlabel(r'$v_{los}$ [km/s]')
axs[0].set_ylabel('Number of galaxies')
axs[1].scatter((red_gals['v_pec']-np.mean(red_gals['v_pec']))/red_gals['sig_clu'], red_gals['R_proj'], color='r', s=20, alpha=0.7)
axs[1].set_xlabel(r'$R_{proj}$ [Mpc]')
axs[1].set_ylabel(r'$(v_{los} – v_{clu}) / \sigma_{clu}$')

axs[0].hist(blue_gals['v_pec'], color='b', alpha=0.9, histtype='step', lw=2, hatch='/', zorder=1)
axs[0].set_xlabel(r'$v_{los}$ [km/s]')
axs[0].set_ylabel('Number of galaxies')
axs[1].scatter((blue_gals['v_pec']-np.mean(blue_gals['v_pec']))/blue_gals['sig_clu'], blue_gals['R_proj'], color='b', s=20, alpha=0.7)
axs[1].set_xlabel(r'$R_{proj}$ [Mpc]')
axs[1].set_ylabel(r'$(v_{los} – v_{clu}) / \sigma_{clu}$')
plt.savefig('redvsblue.pdf', bbox_inches='tight', format='pdf')