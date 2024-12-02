# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:11:26 2024

@author: xGeeRe
"""
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import read_mist_models
from astropy.table import Table
import matplotlib.transforms as transforms
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import extinction
from spectra_utils import EW, NaID_extinction, DiBs_extinction

#Function to plot sources as a scatter plot with density
def density_scatter( x , y, sort = True, bins = 20, ax=None, **kwargs )   :
    '''
    Scatter plot colored by 2d histogram.

    Parameters
    ----------
    x : list of floats
        X axis values.
    y : list of floats
        Y axis values.
    sort : bool, optional
        If True, sort the points by density, so that the densest points are 
        plotted last. The default is True.
    bins : float, optional
        Number of bins for the np.histogram2d. The default is 20.
    ax : Axes object (matplotlib), optional
        Axes where to draw the density plot. The default is None.
    **kwargs :
        Additional parameters for matplotlib.pyplot.scatter.

    Returns
    -------
    out : Scatter plot (matplotlib)
        The density scatter plot.

    '''
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    out = ax.scatter( x, y, c=z, **kwargs )
    #plt.colorbar(label = "Sources per bin", location="left")
        
    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm))
    #cbar.ax.set_ylabel('Density')

    return out


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def CMD(file, table_format= 'ascii.csv', density=False, title=None, MISTdir='MIST', plot=False, savepath=None, ax=None, **kwargs):
    
    table = Table.read(file, format=table_format)
    mag = table['mg0']
    bprp = table['bprp0']
    
    mass_list = ["0005000M", "0010000M", "0015000M", "0020000M", "0025000M", "0030000M", "0035000M", "0040000M", "0045000M", "0050000M", "0055000M", "0060000M", "0065000M", "0070000M", "0075000M", 
            "0080000M", "0085000M", "0090000M", "0095000M", "0100000M", "0105000M", "0110000M", "0115000M", "0120000M", "0125000M", "0130000M", "0135000M", "0140000M", "0145000M", "0150000M", 
            "0155000M", "0160000M", "0165000M", "0170000M", "0175000M", "0180000M", "0185000M", "0190000M", "0195000M", "0200000M"]

    #Cut equations
    y_cut = np.linspace(2, -9, 100)
    def x_cut1(y_cut):
        return (21 / 128) * y_cut + (149 / 386)
    def x_cut2(y_cut):
        return (-7 / 124) * y_cut + (603 / 620)

    plt.ioff()
    if ax is None:
        if density:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(10,8))
        else:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(9,8)) 
    
    
    if density:
        out = density_scatter( np.array(bprp), np.array(mag), bins = 500, ax=ax, **kwargs)
        cb = plt.colorbar(out, pad = 0.02, ax=ax)
        cb.set_label("Sources per bin", fontsize = 25, labelpad = -85)
    else:
        ax.scatter(bprp, mag, **kwargs)
    
    ax.set_xlim(-0.7, 1.5)
    ax.set_ylim(2, -7)
    ax.set_xlabel("$(BP - RP)$ [mag]", fontsize = 19)
    ax.set_ylabel("$M_{G}$ [mag]", fontsize = 19)
    ax.tick_params(labelsize = 16)
    if title is not None:
        ax.set_title(title, fontsize=19)
    
    #Iterators, j is for linestyles only
    i = 3
    j = 0
    c_index = 0
    mass_plots = []
    cmap = cm.plasma
    norm = Normalize(vmin=0, vmax=8)
    lower_bound = 0  # Starting from 20% of the colormap
    upper_bound = 0.85  # Ending at 80% of the colormap
    colors = [cmap(lower_bound + (upper_bound - lower_bound) * norm(ccc)) for ccc in range(9)]
    while i < len(mass_list):
        eepcmd = read_mist_models.EEPCMD(f'{MISTdir}/{mass_list[i]}.track.eep.cmd')
        cmd = eepcmd.eepcmds
        #MS (phase = 0)
        x = cmd["Gaia_BP_EDR3"][cmd["phase"] == 0] - cmd["Gaia_RP_EDR3"][cmd["phase"] == 0]
        y = cmd["Gaia_G_EDR3"][cmd["phase"] == 0]
        #SGB-RGB (phase = 2)
        s = cmd["Gaia_BP_EDR3"][cmd["phase"] == 2] - cmd["Gaia_RP_EDR3"][cmd["phase"] == 2]
        r = cmd["Gaia_G_EDR3"][cmd["phase"] == 2]
    
        if j <= 3:
            #Plot 2 phases on 1 diagram, red:MS, black:SGB-RGB
            ax.plot(x, y, color=colors[c_index], lw=6) #, ls = plot_ls[j]
            l2, = ax.plot(s, r, color=colors[c_index], ls = "--") #, ls = plot_ls[j], label = mass_names[i]
            mass_plots.append(l2)
            c_index += 1
        else:
            ax.plot(x, y, color=colors[c_index], lw=6) #, ls = "--", dashes = dashed_lines[j-4]
            l2, = ax.plot(s, r, color=colors[c_index], ls = "--") #, ls = "--", dashes = dashed_lines[j-4], label = mass_names[i]
            mass_plots.append(l2)
            c_index += 1
        
        j += 1
        if i < 5:
            i += 1
        elif i == 5:
            i += 2
        else:
            i += 4
        
        #In case i is greater than list lenght or if only a set range of masses is wanted
        if i >= 28:
            break
    
    ax.plot(x_cut1(y_cut), y_cut, color = "green")
    ax.plot(x_cut2(y_cut), y_cut, color = "green")
    
    trans = transforms.blended_transform_factory(ax.transData, ax.transData)
    ax.text(-0.15, 1.8, r"2$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[0])
    ax.text(-0.31, 1.3, r"2.5$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[1])
    ax.text(-0.32, 0.7, r"3$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[2])
    
    ax.text(-0.4, 0, r"4$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[3])
    ax.text(-0.5, -0.9, r"6$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[4])
    ax.text(-0.55, -1.6, r"8$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[5])
    
    ax.text(-0.62, -2.1, r"10$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[6])
    ax.text(-0.66, -2.5, r"12$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[7])
    ax.text(-0.68, -3, r"14$M_{\odot}$", transform = trans, fontsize=17, weight='bold', color=colors[8])
    
    #var_legend = plt.legend(handles = var_plots , loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, ncol=1)
    #ax = plt.gca().add_artist(var_legend)
    #mass_legend = plt.legend(handles = mass_plots, loc="upper center", bbox_to_anchor=(0.5, -0.075), fancybox=True, ncol=4)
    
    if savepath is not None:
        plt.savefig(f"{savepath}", bbox_inches = "tight")
    if plot:
        plt.show()
        plt.close()
        
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

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
        xrange = [5890, 5900] # D1
        # xrange = [5884, 5894] # D2
        ews, _, _, _ = EW(filepath, xrange)
        sum_ews = sum(ews)
        _, AV = NaID_extinction(sum_ews, line='D1')
        G_na, BP_RP_na = mag_ext_corr(ide, AV, how='starhorse')
        # DiB
        xrange = [5770, 5790]
        # range = [6604, 6624]
        ews, _, _, _ = EW(filepath, xrange)
        sum_ews = sum(ews)
        _, AV = DiBs_extinction(sum_ews, line='5780')
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
    ax.set_xlim(-1.7, 1.6)
    ax.set_ylim(2, -8)
    plt.show()
    plt.close()