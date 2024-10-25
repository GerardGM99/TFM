# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:53:43 2024

@author: xGeeRe
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import spectra_utils as su
import lightcurve_utils as lu
import phot_utils as pu
import coord_utils as cu
from astropy.table import Table
import Finker_script as fink
import numpy as np
import pandas as pd
import matplotlib.transforms as transforms
from lightcurve_utils import remove_outliers, bin_lightcurve
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os
from finder_chart import draw_image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from astropy.time import Time

rcParams['font.family'] = 'serif'

def original_lc(file_name, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, 
                rang=None, plot=False, savefig=True, ax=None, add_secondary_xaxis=True):

    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF', 'zg'):'g',
        ('ZTF', 'zr'):'r',
        ('ZTF', 'zi'):'goldenrod',
        ('IRSA_ZTF', 'zg'):'g',
        ('IRSA_ZTF', 'zr'):'r',
        ('IRSA_ZTF', 'zi'):'gold',
        ('ASAS-SN', 'V'):'darkcyan',
        ('ASAS-SN', 'g'):'blue',
        ('ATLAS', 'o'):'orange',
        ('ATLAS', 'c'):'cyan',
        ('NEOWISE', 'W1'):'darkred',
        ('NEOWISE', 'W2'):'slategray',
        ('BLACKGEM', 'u'):'purple',
        ('BLACKGEM', 'g'):'skyblue',
        ('BLACKGEM', 'r'):'orange',
        ('BLACKGEM', 'i'):'firebrick',
        ('BLACKGEM', 'z'):'sienna',
        ('BLACKGEM', 'q'):'black',
        ('MeerLICHT', 'u'):'purple',
        ('MeerLICHT', 'g'):'skyblue',
        ('MeerLICHT', 'r'):'orange',
        ('MeerLICHT', 'i'):'firebrick',
        ('MeerLICHT', 'z'):'sienna',
        ('MeerLICHT', 'q'):'black',
        ('TESS', 'QLP'):'magenta',
        ('TESS', 'SPOC'):'magenta',
        ('TESS', 'TASOC'):'magenta',
        ('TESS', 'CDIPS'):'magenta',
        ('TESS', 'TGLC'):'magenta',
        ('TESS', 'GSFC-ELEANOR-LITE'):'magenta'}
    
    plt.ioff()
    file = Table.read(file_name, format='ascii.csv')
    if len(file) > 4:
        ins = file['inst'][0]
        source_name = file['name'][0]
        out_path = f'{ins}_plots'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            
        if ins == 'NEOWISE':
            ms = 6
            cs = 4
            ew = 4
        else:
            ms = 3
            cs = 2
            ew = 2
            
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
            fig.set_size_inches(9, 5.5)
            ax.grid(alpha=0.5)
            
        available_filters = sorted(set(file['filter']))
        for band in available_filters:
            if y_ax == 'mag':
                if ins == 'NEOWISE':
                    t_observed = np.array(file["mjd"][file['filter'] == band])
                else:
                    t_observed = np.array(file["bjd"][file['filter'] == band])
                y_observed = np.array(file['mag'][file['filter'] == band])
                uncert = np.array(file["magerr"][file['filter'] == band])
                    
                if binsize > 0:
                    t_observed, y_observed, t_err, uncert = bin_lightcurve(t_observed, y_observed, yerr=uncert, 
                                                                            binsize=binsize, mode="average")
                else:
                    t_err = None
                        
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    if binsize > 0:
                        t_err = t_err[mask]
                    
                plot_color = color_dict.get((ins, band), 'black')
                if ins=='NEOWISE':
                    ax.plot(t_observed - ref_mjd, y_observed, linestyle='--', color=plot_color, alpha=0.5)
                ax.errorbar(t_observed - ref_mjd, y_observed, xerr=t_err, yerr=uncert, color=plot_color, label=band, 
                            fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Magnitude", family="serif", fontsize=19)
                    
            if y_ax == 'flux':
                sigma = remove_outliers(file['flux'][file['filter'] == band], method=outliers)
                    
                t_observed = np.array(file["bjd"][file['filter'] == band])[sigma]
                y_observed = np.array(file['flux'][file['filter'] == band])[sigma]
                uncert = np.array(file["fluxerr"][file['filter'] == band])[sigma]
                
                if rang is not None:
                    mask = (t_observed > rang[0]) & (t_observed < rang[1])
                    t_observed = t_observed[mask]
                    y_observed = y_observed[mask]
                    uncert = uncert[mask]
                    
                plot_color = color_dict.get((ins, band), 'black')
                ax.errorbar(t_observed - ref_mjd, y_observed, yerr=uncert, color=plot_color, label=band, 
                            fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Flux", family="serif", fontsize=16)
                    
        ax.set_title(f"{source_name}", weight="bold") 
        ax.set_xlabel(f"MJD - {ref_mjd} [days]", family="serif", fontsize=19)
        ax.tick_params(which='major', width=2, direction='out')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, direction='out')
        ax.tick_params(labelsize=18)
        ax.minorticks_on()
        ax.invert_yaxis()
        xmin, xmax = ax.get_xlim()    
        
        if add_secondary_xaxis:
            def xconv(x):
                tyear = Time(x + ref_mjd, format="mjd")
                return tyear.jyear 
            
            xmin2 = xconv(xmin)
            xmax2 = xconv(xmax)
            
            ay2 = ax.twiny()
            ay2.minorticks_on()
            ay2.tick_params(which='major', width=2, direction='out')
            ay2.tick_params(which='major', length=7)
            ay2.tick_params(which='minor', length=4, direction='out')
            ay2.tick_params(labelsize=18)
            ay2.set_xlim([xmin2, xmax2])
            ay2.set_xlabel('Year', fontsize=19)
            ay2.ticklabel_format(useOffset=False)
            
        colors_for_filters = [color_dict.get((ins, filt), None) for filt in available_filters]
                
        ax.legend(loc="upper right", ncols=len(available_filters), labelcolor=colors_for_filters, 
                  shadow=True, columnspacing=0.8, handletextpad=0.5, handlelength=1, markerscale=1.5, 
                  prop={"weight": "bold", 'size':15})
        
        if savefig:
            plt.savefig(f'{out_path}/{source_name}.png', bbox_inches="tight", format="png")
        
        if plot:
            plt.show()
        
        if ax is None:
            plt.close()

def stacked_og_lc(name, ins, facecolor):
    
    # Create a new figure with GridSpec
    if len(ins)==1:
        fig = plt.figure(constrained_layout=True, figsize=(7.3,7.3), facecolor=facecolor)
    else:
        fig = plt.figure(constrained_layout=True, figsize=(7.3,3.5*len(ins)), facecolor=facecolor)
    gs = gridspec.GridSpec(len(ins), 1, figure=fig, hspace=0)  # hspace=0 to remove separation between plots
    
    # Define the axes, sharing the x-axis
    if len(ins)==1:
        ax1 = fig.add_subplot(gs[0])
        axes=[ax1]
    elif len(ins)==2:
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        axes=[ax1, ax2]
    elif len(ins)==3:
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        axes=[ax1, ax2, ax3]
    
    # Plot the light curves on the respective axes
    for i, instrument in enumerate(zip(ins,axes)):
        if i == 0:
            secondary = True
        else:
            secondary = False
            
        original_lc(f'{instrument[0]}_lightcurves_std/{name}.csv', ax=instrument[1], plot=False, 
                    savefig=False, add_secondary_xaxis=secondary)
    
    for j, ax in enumerate(axes):
        ax.set_title('')
        if (len(axes)>1 and j==0) or (len(axes)>2 and j==1):
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False, direction='in')
            ax.tick_params(which='minor', length=4, direction='in')
        if len(axes)>2 and j==1:
            ax.set_xlabel('')
            
    plt.tight_layout()
    plt.savefig(f'cool_plots/stacked_lcs/{name}.png', bbox_inches="tight", format="png")
    plt.close()

def lc_folding(name, time, y, uncert, best_freq, ax, t_start=None, cycles=1, outliers='eb', yunits='mag'):
    
    # Remove outliers
    sigma = remove_outliers(y, method=outliers)
    
    time = np.array(time)[sigma]
    y = np.array(y)[sigma]
    if uncert is not None:
        uncert = np.array(uncert)[sigma]
    
    # Set initial time
    if t_start is None:
        min_index = np.argmax(y)
        time = time - time.iloc[min_index]
    else:
        time = time - t_start
    
    name = name.split('_')[0]
    
    # Plot folded light curve
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.errorbar((time * best_freq) % cycles, y,yerr=uncert, fmt='o', ecolor='black', capsize=2, elinewidth=2,
                 markersize=2, markeredgewidth=0.5, alpha=0.8, color='black',zorder=0, mec='black')
    ax.set_xlabel('Phase', fontsize=19)
    if yunits=='mag':
        ax.invert_yaxis()
        ax.set_ylabel(r'Magnitude', fontsize=19)
    elif yunits=='flux':
        ax.set_ylabel(r'Flux', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # plt.suptitle(name, fontsize=20, weight='bold')
    if ((24*60)/best_freq)<60:
        ax.set_title(f'Freq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {(24*60) / best_freq:.3f}' + r'$ \, min$',
                     fontsize=20)
    elif ((((24*60)/best_freq)>60) and (((24*60)/best_freq)<(24*60))):
        ax.set_title(f'Freq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {24 / best_freq:.3f}' + r'$ \, hr$',
                     fontsize=20)
    else:
        ax.set_title(f'Freq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {1/best_freq:.3f}' + r'$ \, d$',
                     fontsize=20)
   


def lc_combined(name, t_list, y_list, y_list_err, filt_list, best_freq, t_start=None, cycles=2, outliers='eb', facecolor=None):
    
    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF, zg'):'g',
        ('ZTF, zr'):'r',
        ('ZTF, zi'):'goldenrod',
        ('IRSA_ZTF, zg'):'g',
        ('IRSA_ZTF, zr'):'r',
        ('IRSA_ZTF, zi'):'goldenrod',
        ('ASAS-SN, V'):'darkcyan',
        ('ASAS-SN, g'):'blue',
        ('ATLAS, o'):'orange',
        ('ATLAS, c'):'cyan',
        ('NEOWISE, W1'):'darkred',
        ('NEOWISE, W2'):'slategray',
        ('BLACKGEM, u'):'purple',
        ('BLACKGEM, g'):'skyblue',
        ('BLACKGEM, r'):'orange',
        ('BLACKGEM, i'):'firebrick',
        ('BLACKGEM, z'):'sienna',
        ('BLACKGEM, q'):'black',
        ('TESS, QLP'):'magenta',
        ('TESS, SPOC'):'magenta',
        ('TESS, TASOC'):'magenta',
        ('TESS, CDIPS'):'magenta',
        ('TESS, TGLC'):'magenta',
        ('TESS, GSFC-ELEANOR-LITE'):'magenta'}
    
    if len(t_list)==1:
        fig = plt.figure(constrained_layout=True, figsize=(7.3,7.3), facecolor=facecolor)
    else:
        fig = plt.figure(figsize=(7.3,3.5*len(t_list)), facecolor=facecolor)
    height_ratios = np.repeat(1, len(t_list))
    gs = gridspec.GridSpec(len(t_list), 1, height_ratios=height_ratios)
    
    if len(t_list)==1:
        ax1 = fig.add_subplot(gs[0])
        lc_folding(name, t_list[0], y_list[0], y_list_err[0], best_freq, ax1, t_start=t_start, cycles=cycles, outliers=outliers, yunits='mag')
        # Change color depending on ins and filter
        errorbars = ax1.get_children()
        plot_color = color_dict.get(filt_list[0], 'black')
        for j in range(4):
            errorbars[j].set_color(plot_color)
        trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
        ax1.text(0.02,0.93, filt_list[0], fontsize=18, transform = trans, style='italic')
    else:
        for i in range(len(t_list)):
            if filt_list[i].split(',')[0]=='TESS':
                yunits='flux'
            else:
                yunits='mag'
                
            if i==0:
                ax1 = fig.add_subplot(gs[i])
                lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax1, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
                # Change color depending on ins and filter
                errorbars = ax1.get_children()
                plot_color = color_dict.get(filt_list[i], 'black')
                for j in range(4):
                    errorbars[j].set_color(plot_color)
                ax1.tick_params(axis='x', labelbottom=False, direction='in')
                ax1.set_xlabel('')
                trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
                ax1.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
            elif i==max(range(len(t_list))):
                ax = fig.add_subplot(gs[i], sharex=ax1)
                lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
                # Change color depending on ins and filter
                errorbars = ax.get_children()
                plot_color = color_dict.get(filt_list[i], 'black')
                for j in range(4):
                    errorbars[j].set_color(plot_color)
                ax.set_title('')
                trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
                ax.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
            else:
                ax = fig.add_subplot(gs[i], sharex=ax1)
                lc_folding(name, t_list[i], y_list[i], y_list_err[i], best_freq, ax, t_start=t_start, cycles=cycles, outliers=outliers, yunits=yunits)
                # Change color depending on ins and filter
                errorbars = ax.get_children()
                plot_color = color_dict.get(filt_list[i], 'black')
                for j in range(4):
                    errorbars[j].set_color(plot_color)
                ax.set_title('')
                ax.tick_params(axis='x', labelbottom=False, direction='in')
                ax.set_xlabel('')
                trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
                ax.text(0.02,0.93, filt_list[i], fontsize=18, transform = trans, style='italic')
        
    gs.update(hspace=0)
    
    
def bulk_combine(name, instruments, best_freq, cycles=2, outliers='eb', facecolor=None):

    times=[]
    mags=[]
    mag_errs=[]
    filts=[]
    for ins in instruments:
        if ins!='TESS':
            nams = name.split('_')[0]
            table = pd.read_csv(f'{ins}_lightcurves_std/{nams}.csv')
        else:
            table = pd.read_csv(f'{ins}_lightcurves_std/{name}.csv')
        bands=list(set(table['filter']))
        if 'ZTF' in ins:
            desired_order = ['zg', 'zr', 'zi']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
        elif 'ATLAS' in ins:
            desired_order = ['c', 'o']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
        elif 'ASAS' in ins:
            desired_order = ['g', 'V']
            def sort_key(band):
                return desired_order.index(band)
            bands.sort(key=sort_key)
            
        for i, band in enumerate(bands):
            t=table['bjd'].loc[table['filter']==band]
            if ins=='TESS':
                y=table['flux'].loc[table['filter']==band]
                yerr=table['fluxerr'].loc[table['filter']==band]
            else:
                y=table['mag'].loc[table['filter']==band]
                yerr=table['magerr'].loc[table['filter']==band]
            if i == 0:
                min_index = np.argmax(y)
                t_start = t.iloc[min_index]
            filt = table['inst'][0] + ', ' + band
            times.append(t)
            mags.append(y)
            mag_errs.append(yerr)
            filts.append(filt)

    lc_combined(name, times, mags, mag_errs, filts, best_freq, t_start=t_start, cycles=cycles, outliers=outliers, facecolor=facecolor)

    plt.tight_layout()
    plt.savefig(f'cool_plots/folds/{name}.png', bbox_inches="tight", format="png")
    plt.close()
    
#################################################

def dashboard(name):
    
    titles_fontsize = 20
    dash_titles = 30
    
    tiers_tab = Table.read('data/new_TFM_table.csv', format='ascii.csv')
    tier = tiers_tab['Tier'][tiers_tab['Gaia source ID']==int(name)]
    def lighten_color(color, factor=0.5):
        # Blend the color with white
        return tuple(min(1, c * (1 - factor) + factor) for c in color)

    if tier == 'Diamond':
        tier_name='Diamond'
        base_color = (127/255, 255/255, 212/255)
        facecolor = lighten_color(base_color)
    elif tier == 'Gold':
        tier_name='Gold'
        base_color = (240/255, 230/255, 140/255)
        facecolor = lighten_color(base_color)
    elif tier == 'Silver':
        tier_name='Silver'
        facecolor = (0.9, 0.9, 0.9)  # Silver does not use alpha, so no change needed
    elif tier == 'Bronze':
        tier_name='Bronze'
        base_color = (205/255, 127/255, 50/255)
        facecolor = lighten_color(base_color)
        
    table = Table.read('data/70_targets_extended.csv', format='ascii.csv')
    ztf = os.listdir('IRSA_ZTF_lightcurves_std')
    for i, fil in enumerate(ztf):
        ztf[i] = fil.split('.')[0]
    asas = os.listdir('ASAS-SN_lightcurves_std')
    for i, fil in enumerate(asas):
        asas[i] = fil.split('.')[0]
    atlas = os.listdir('ATLAS_lightcurves_std')
    for i, fil in enumerate(atlas):
        atlas[i] = fil.split('.')[0]
    
    exceptions = ['2175699216614191360']
    exceptions2 = ['4076568861833452160']
    if name in ztf and name not in exceptions and name not in exceptions2:
        tubla = pd.read_csv(f'IRSA_ZTF_lightcurves_std/{name}.csv')
        bands = list(set(tubla['filter']))
        if 'zg' in bands:
            band = 'zg'
        else:
            band = 'zr'
    elif name in atlas and name not in exceptions2:
        tubla = pd.read_csv(f'ATLAS_lightcurves_std/{name}.csv')
        band = 'c'
    elif name in asas:
        tubla = pd.read_csv(f'ASAS-SN_lightcurves_std/{name}.csv')
        band = 'g'
    
    bands_dict = {'5524022735225482624':'o', '2206767944881247744':'zr', '4076568861833452160':'V', 
                  '2006088484204609408':'zr', '2061252975440642816':'zr', 
                  '5311969857556479616':'o', '6228685649971375616':'zr', '2175699216614191360':'o',
                  '5326288831829996416':'o', '5966574133883852416':'o'}
    
    if name in bands_dict:
        band = bands_dict.get(name)
    
    inss = []
    if name in ztf:
        inss.append('IRSA_ZTF')
    elif name in atlas and name in asas:
        inss.append('ATLAS')
        inss.append('ASAS-SN')
    elif name in atlas and name not in asas:
        inss.append('ATLAS')
    elif name not in atlas and name in asas:
        inss.append('ASAS-SN')
        
    inss2 = []
    if name in ztf:
        inss2.append('IRSA_ZTF')
    if name in atlas:
        inss2.append('ATLAS')
    if name in asas:
        inss2.append('ASAS-SN')
    
    mask = tubla['filter'] == band
    t = np.array(tubla['bjd'][mask])
    y = np.array(tubla['mag'][mask])
    yerr = np.array(tubla['magerr'][mask])
    
    fig = plt.figure(figsize=(24,15.75), layout='constrained', facecolor=facecolor)
    outer_gs = GridSpec(nrows=3, ncols=4, figure=fig,
                        height_ratios=[1,0.9,1.1],
                        width_ratios=[1.1,1.3,0.8,0.8])
    
    
    left_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[:-1, 0], height_ratios=[1.1, 0.9])
    
    # Bottom row with different column width ratios
    bottom_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[2, :], width_ratios=[2, 1, 1, 1])
    
    ax0 = fig.add_subplot(left_gs[0, 0]) #cutout
    ax1 = fig.add_subplot(left_gs[1, 0]) #cmd
    ax2 = fig.add_subplot(outer_gs[:-1, 1]) #(folded) lcs
    ax3 = fig.add_subplot(outer_gs[0, 2]) #finker left
    ax5 = fig.add_subplot(outer_gs[0, 3]) #finker right
    ax4 = fig.add_subplot(outer_gs[1, 2:4]) #neowise lc
    
    # Bottom row subplots with adjusted widths
    ax7 = fig.add_subplot(bottom_gs[0, :2]) #spectrum
    ax8 = fig.add_subplot(bottom_gs[0, 2:4]) #BPRP
    
    mask = table['source_id']==int(name)
    
    # CUTOUT
    cutout_images = os.listdir('cool_plots/images/color')
    compare_list = []
    for ima in cutout_images:
        compare_list.append(ima.split('_')[0])
    if name in compare_list and name!='5866345647515558400':
        size= 160
        cu.draw_image(table['ra'][mask][0], table['dec'][mask][0], name, size, plot=False, save=False, ax=ax0)
        ax0.set_xlabel('%.1f\''%(size*0.25/60), fontsize = titles_fontsize)
        ax0.set_ylabel('%.1f\''%(size*0.25/60), fontsize = titles_fontsize)
        ax0.set_title('PANSTARS cutout', fontsize=titles_fontsize, weight='bold')
    elif name=='5866345647515558400':
        imagen = imread("cool_plots/images/Color/5866345647515558400_EPIC.png")
        ax0.imshow(imagen, aspect='auto')
        ax0.axis('off')
        ax0.set_title('EPIC cutout', fontsize=titles_fontsize, weight='bold')
    elif name not in compare_list and name not in ['5866474526572151936', '5880159842877908352']:
        rad= 20/3600
        ax0 = draw_image(table['ra'][mask][0], table['dec'][mask][0], name, rad, save=False, plot=False, ax=ax0)
        ax0.set_xlabel('%.1f\'' % (rad * 60 * 2), fontsize = titles_fontsize)
        ax0.set_ylabel('%.1f\'' % (rad * 60 * 2), fontsize = titles_fontsize)
        ax0.set_title('DSS cutout', fontsize=titles_fontsize, weight='bold')
    elif name in ['5866474526572151936', '5880159842877908352']:
        ax0.axis('off')
    
    # CMD
    pu.CMD('data/70_targets_extended.csv', s=50, color='k', alpha=0.5, ax=ax1)
    ax1.scatter(table['bprp0'][mask], table['mg0'][mask], s=300, color='r', marker='*')
    ax1.set_xlabel("$(BP - RP)_{SH}$ [mag]", fontsize = titles_fontsize)
    ax1.set_ylabel("$M_{G,SH}$ [mag]", fontsize = titles_fontsize)
    ax1.tick_params(labelsize = titles_fontsize)
    ax1.set_title('Gaia CMD', fontsize=titles_fontsize, weight='bold')
    
    # FOLDED LIGHT CURVES
    dictionary = {'2166378312964576256':72.427805, '5524022735225482624':7.988562, '2060841448854265216':60.545008,
                  '2175699216614191360':4.769977, '2206767944881247744':2*4.625127/24, '5880159842877908352':63.541328, 
                  '460686648965862528':3.7606215/24, '5965503866703572480':7.321043/24, '4076568861833452160':56.90033, 
                  '5323384162646755712':2.792011, '2006088484204609408':2.5960645, '2061252975440642816':2.139183, 
                  '6123873398383875456':19.0433664/24, '5311969857556479616':13.7444267, '6228685649971375616':1.273980,
                  '1870955515858422656':46.211840}
    if name in dictionary:
        period = dictionary.get(name)
        if name=='4076568861833452160':
            inss=['IRSA_ZTF', 'ASAS-SN']
        bulk_combine(name, inss, 1/(period), cycles=2, outliers='eb', facecolor=facecolor)
        image2 = imread(f"cool_plots/folds/{name}.png")
        try:
            if len(bands)==1:
                ax2.imshow(image2)#, aspect='auto')
            else:
                ax2.imshow(image2, aspect='auto')
        except:
            ax2.imshow(image2, aspect='auto')
        # ax2.set_title('Folded light curve', fontsize=titles_fontsize, weight='bold')
        ax2.axis('off')
    else:
        # ins=['ATLAS']
        stacked_og_lc(name, inss2, facecolor=facecolor) 
        image2 = imread(f"cool_plots/stacked_lcs/{name}.png")
        if len(inss2)==1:
            ax2.imshow(image2)#, aspect='auto')
        else:
            ax2.imshow(image2, aspect='auto')
        # ax2.set_title('Light curve', fontsize=titles_fontsize, weight='bold')
        ax2.axis('off')
    
    # FINKER OUTPUT
    freq = np.linspace(0.001,7,28000)
    _ = fink.Finker_mag(t, y, yerr, freq, show_plot=False, calc_error=False, ax1=ax3, ax2=ax5)
    ax3.set_title("FINKER's frequency search", fontsize=titles_fontsize, weight='bold')
    ax5.set_title("FINKER'S folded LC", fontsize=titles_fontsize, weight='bold')
    
    # NEOWISE LIGHT CURVE
    lu.plot_ind_lightcurves(f'NEOWISE_lightcurves_std/{name}.csv',plot=False, savefig=False, ax=ax4, binsize=50)
    ax4.set_title('NEOWISE ligth curve', fontsize=titles_fontsize, weight='bold')
    
    # SPECTRUM (CAFOS/LAMOST)
    spec_dict = {'2030965725082217088':('0200',None), '2074693061975359232':('0201',None),
                 '2164630463117114496':('0202',2e-13), '2206767944881247744':('0203',5e-13),
                 '1870955515858422656':('0204',0.9e-13), '2013187240507011456':('0205',3e-13),
                 '2060841448854265216':('0206',0.4e-12), '2166378312964576256':('0207',None),
                 '2200433413577635840':('0208',None)}
    fies = ['4076568861833452160', '4299904519833646080']
    if name in spec_dict:
        unit = spec_dict.get(name)[0]
        spec_lim = spec_dict.get(name)[1]
        directory = 'data/cafos_spectra'
        su.cafos_spectra(f'{directory}/spectra1D_dswfz_uniB_{unit}.txt', 'data/TFM_table.csv', dered='fitz',
                          lines_file='data/spectral_lines.txt', plot=False, ax=ax7)
        ax7.set_title('CAFOS dereddened spectrum', fontsize=titles_fontsize, weight='bold')
        if spec_lim is not None:
            ax7.set_ylim(top=spec_lim)
    elif name=='187219239343050880':
        directory = 'data/lamost_spectra'
        x1, y1 = su.lamost_spectra(f'{directory}/spec-55977-GAC_080N37_V1_sp03-241.fits', 'data/TFM_table.csv', 
                                   dered='fitz', lines_file='data/spectral_lines.txt', plot=False, ax=ax7)
        x2, y2 = su.lamost_spectra(f'{directory}/spec-56687-GAC080N37V1_sp03-241.fits', 'data/TFM_table.csv', dered='fitz',
                                   plot=False, ax=ax7, color='blue')
        x3, y3 = su.lamost_spectra(f'{directory}/spec-56948-HD052351N362354B_sp14-213.fits', 'data/TFM_table.csv', dered='fitz',
                                   plot=False, ax=ax7, color='orange')
        
        inset_ax = inset_axes(ax7, width="100%", height="100%", 
                      bbox_to_anchor=(7100, 10000, 800, 7000), 
                      bbox_transform=ax7.transData, 
                      loc='upper left')
        inset_ax.plot(x1, y1/np.mean(y1[(x1>6554)*(x1<6575)]), color='k')
        inset_ax.plot(x2, y2/np.mean(y2[(x2>6554)*(x2<6575)]), color='blue')
        inset_ax.plot(x3, y3/np.mean(y3[(x3>6554)*(x3<6575)]), color='orange')
        inset_ax.set_xlim(6554, 6575)  # Assuming Halpha is around this range
        inset_ax.set_ylim(0.9, 1.15)
        inset_ax.axvline(6562.8, ls='dashed', color='r')
        inset_ax.text(6563.5, 1.12, "H$\\alpha$", fontdict={'fontsize': 12})
        inset_ax.set_ylabel('Normalized flux', fontsize=12)
        inset_ax.set_title('Zoom around H$\\alpha$', fontsize=12)
        
        ax7.text(7000, 7000, '19-02-2012', fontsize=15, color='k')
        ax7.text(7000, 4000, '29-01-2014', fontsize=15, color='blue')
        ax7.text(7000, 1100, '17-10-2014', fontsize=15, color='orange')        
        ax7.set_title('LAMOST dereddened spectrum', fontsize=titles_fontsize, weight='bold')
        ax7.set_ylim(top=18000, bottom=0)
    elif name=='6123873398383875456':
        directory = 'data/HERMES_spectra'
        spec = pd.read_csv(f'{directory}/{name}_11-5.txt', sep=' ')
        su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[6520,6610], #Av=5.7289376,
                    units=['Angstrom','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'], lines_file='data/spectral_lines.txt',
                    priority=['1'], plot=False, ax=ax7, color='k')
        spec = pd.read_csv(f'{directory}/{name}_13-5.txt', sep=' ')
        su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[6520,6610], #Av=5.7289376,
                    units=['Angstrom','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'], lines_file='data/spectral_lines.txt',
                    priority=['1', '2'], plot=False, ax=ax7, color='blue', alpha=0.5)
        
        ax7.text(6530, 4.5e-16, '11-05-2024', fontsize=15, color='k')
        ax7.text(6530, 4e-16, '13-05-2012', fontsize=15, color='blue')      
        ax7.set_title(r'HERMES high resolution spectrum (around $H\alpha$)', fontsize=titles_fontsize, weight='bold')
    elif name=='3444168325163139840':
        directory = 'data/lamost_spectra'
        su.lamost_spectra(f'{directory}/med-58149-HIP265740401_sp09-191.fits', 'data/TFM_table.csv',
                          lines_file='data/spectral_lines.txt', plot=False, ax=ax7)
        ax7.set_title('LAMOST medium resolution spectrum', fontsize=titles_fontsize, weight='bold')
        # ax7.set_ylim(bottom=1000, top=6000)
        # ax7.set_xlim(left=6250, right=6850)
        ax7.set_ylim(bottom=5000, top=37000)
        ax7.set_xlim(left=6270, right=6900)
    elif name in fies:
        directory = 'data/FIES-M_spectra'
        spec = pd.read_csv(f'{directory}/{name}.txt', sep=' ')
        su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[6500,6610], #Av=5.7289376,
                    units=['Angstrom','W nm$^{-1}$ m$^{-2}$'], lines_file='data/spectral_lines.txt',
                    priority=['1', '2'], plot=False, ax=ax7, color='k')
        ax7.set_title(r'FIES medium resolution spectrum (around $H\alpha$)', fontsize=titles_fontsize, weight='bold')
    else:
        ax7.axis('off')
    
    # GAIA BPRP SPECTRUM
    su.Gaia_XP([name], ax=ax8)
    ax8.set_title('Gaia BPRP spectrum', fontsize=titles_fontsize, weight='bold')
    
    # DASHBOARD TITLE
    if name in ['4054010697162430592', '2083649030845658624', '5866345647515558400']:
        fig.suptitle('Gaia DR3 '+name+' (hard X-ray emitter)', weight='bold', fontsize=dash_titles, x=0.02, horizontalalignment='left')
    elif name == '473575777103322496':
        fig.suptitle('Gaia DR3 '+name+' (UV excess)', weight='bold', fontsize=dash_titles, x=0.02, horizontalalignment='left')
    else:
        fig.suptitle('Gaia DR3 '+name, weight='bold', fontsize=dash_titles, x=0.02, horizontalalignment='left')
    fig.text(0.98, 0.98, f'{tier_name} Tier', weight='bold', fontsize=dash_titles, horizontalalignment='right')
    
    plt.savefig(f'cool_plots/dashboards/{name}.png', bbox_inches="tight", format="png")
    plt.savefig(f'cool_plots/dashboards/pdfs/{name}.pdf', bbox_inches="tight", format="pdf")
    # plt.show()

# tabla = Table.read('data/70_targets.csv', format='ascii.csv')
# for name in tabla['DR3_source_id']:
#     dashboard(str(name))

# tal = ['4076568861833452160', '4299904519833646080', '6123873398383875456'] #fies, fies, hermes
# for name in tal:
#     dashboard(name)

dashboard('4076568861833452160')

# lamost med: 
#   'med-58149-HIP265740401_sp09-191.fits' corresponds to 3444168325163139840
#   'med-58440-NT222300N571704C01_sp11-073.fits' corresponds to 2200433413577635840

# cutouts:
    # NO HI HA: 5866474526572151936, 5880159842877908352
    # EPIC: 5866345647515558400

########################

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# newton = pd.read_csv('data/x-ray/4XMM-DR13/NXSA-Results-1714938050811.csv')
# data = newton[newton['EPIC_SOURCE_CAT.OBSERVATION_ID']==201200201]
# ep2 = data['EPIC_SOURCE_CAT.EP_2_FLUX']
# ep3 = data['EPIC_SOURCE_CAT.EP_3_FLUX']
# ep4 = data['EPIC_SOURCE_CAT.EP_4_FLUX']
# ep5 = data['EPIC_SOURCE_CAT.EP_5_FLUX']
# # ep8 = data['EPIC_SOURCE_CAT.EP_8_FLUX']
# fluxes = [ep2, ep3, ep4, ep5]
# wavelenghts = [16.531, 8.2656, 3.8149, 1.5028]
# newton_df = pd.DataFrame({'wavelength':wavelenghts, 'flux':fluxes})

# vosa = pd.read_csv('data/SED/vosa_results_77360/objects/2083649030845658624/sed/2083649030845658624_sed.csv')

# plt.figure(figsize=(8,8))
# plt.scatter(vosa['Wavelength'], vosa['Wavelength']*vosa['Flux'], s=10, color='r')
# plt.scatter(newton_df['wavelength'], newton_df['wavelength']*newton_df['flux'], s=10, color='r')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
