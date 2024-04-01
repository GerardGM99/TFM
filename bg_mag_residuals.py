# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:26:32 2024

@author: xGeeRe
"""

from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

file = Table.read('data/Nadia_south_table.csv', format='ascii.csv')

mask = (file['FLAGS_MASK'] == 0) & (file['SNR_OPT'] > 2) & (file['MAG_OPT'] > file['BACKGROUND'])

df = file[mask].to_pandas()

for source in set(df['DR3_source_id']):
    s_m = df['DR3_source_id'] == source
    
    fig = plt.figure(figsize=(14,21))
    gs = gridspec.GridSpec(6, 1, height_ratios=[2, 1, 1, 1, 1, 1])
    #plt.title(f'{source}', fontsize=18, weight='bold')
    
    ax1 = fig.add_subplot(gs[0])
    s=40
    ax1.scatter(df['MJD-OBS'][s_m], df['MAG_OPT'][s_m], s=s, marker='o',  label='Opt')
    ax1.scatter(df['MJD-OBS'][s_m], df['MAG_APER_R1.5xFWHM'][s_m], s=s, marker='X', label='Aper 1.5x')
    ax1.scatter(df['MJD-OBS'][s_m], df['MAG_APER_R5xFWHM'][s_m], s=s, marker='^', label='Aper 5x')
    ax1.scatter(df['MJD-OBS'][s_m], df['MAG_ZOGY_PLUSREF'][s_m], s=s, marker='D', label='Plus ref')
    ax1.invert_yaxis()
    ax1.set_title(f'{source}', fontsize=18, weight='bold')
    plt.ylabel("Magnitude", fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.90))
    
    s=40
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.scatter(df['MJD-OBS'][s_m],df['MAG_OPT'][s_m]-df['MAG_APER_R1.5xFWHM'][s_m], s = s, label='Opt-Aper 1.5x')
    ax2.scatter(df['MJD-OBS'][s_m],df['MAG_OPT'][s_m]-df['MAG_APER_R5xFWHM'][s_m], s = s, label='Opt-Aper 5x')
    ax2.scatter(df['MJD-OBS'][s_m],df['MAG_OPT'][s_m]-df['MAG_ZOGY_PLUSREF'][s_m], s = s, label='Opt-Plus ref')
    plt.ylabel("Residuals (mag)", fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.84))
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R1.5xFWHM'][s_m]-df['MAG_OPT'][s_m], s = s, label='Aper 1.5x-Opt')
    ax3.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R1.5xFWHM'][s_m]-df['MAG_APER_R5xFWHM'][s_m], s = s, label='Aper 1.5x-Aper 5x')
    ax3.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R1.5xFWHM'][s_m]-df['MAG_ZOGY_PLUSREF'][s_m], s = s, label='Aper 1.5x-Plus ref')
    plt.ylabel("Residuals (mag)", fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.84))
    
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R5xFWHM'][s_m]-df['MAG_OPT'][s_m], s = s, label='Aper 5x-Opt')
    ax4.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R5xFWHM'][s_m]-df['MAG_APER_R1.5xFWHM'][s_m], s = s, label='Aper 5x-Aper 1.5x')
    ax4.scatter(df['MJD-OBS'][s_m],df['MAG_APER_R5xFWHM'][s_m]-df['MAG_ZOGY_PLUSREF'][s_m], s = s, label='Aper 5x-Plus ref')
    plt.ylabel("Residuals (mag)", fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.84))
    
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.scatter(df['MJD-OBS'][s_m],df['MAG_ZOGY_PLUSREF'][s_m]-df['MAG_OPT'][s_m], s = s, label='Plus ref-Opt')
    ax5.scatter(df['MJD-OBS'][s_m],df['MAG_ZOGY_PLUSREF'][s_m]-df['MAG_APER_R1.5xFWHM'][s_m], s = s, label='Plus ref-Aper 1.5x')
    ax5.scatter(df['MJD-OBS'][s_m],df['MAG_ZOGY_PLUSREF'][s_m]-df['MAG_APER_R5xFWHM'][s_m], s = s, label='Plus ref-Aper 5x')
    plt.ylabel("Residuals (mag)", fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.84))
    
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.scatter(df['MJD-OBS'][s_m],df['S-SEEING'][s_m], s = s)
    plt.ylabel("Seeing (arcsec)", fontsize=16)
    plt.xlabel("MJD (days)", fontsize=18)
    
    fig.subplots_adjust(hspace=0)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis='x', labelbottom=False)
        
    #plt.show()
    plt.savefig(f'data/bg_residuals/{source}.png', bbox_inches = "tight", format = "png")
    