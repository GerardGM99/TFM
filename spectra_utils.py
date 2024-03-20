# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:49:06 2024

@author: Gerard Garcia
"""

from gaiaxpy import convert, calibrate
from gaiaxpy import plot_spectra
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np

def Gaia_XP(id_list, out_path=None, plot=False):
    
    #Balmer lines (nm)
    H_alfa = 656.3
    H_beta = 486.1
    H_gamma = 434.1
    
    calibrated_spectra, sampling = calibrate(id_list)

    for i in range(len(calibrated_spectra)):
        source = calibrated_spectra.iloc[[i]]
        ide = source['source_id'].iloc[0]
        plt.figure(figsize=(14,6))
        plt.errorbar(sampling, np.array(source['flux'])[0], yerr=np.array(source['flux_error'])[0], 
                     fmt=".-", color="k", label = "DR3")
        plt.axvline(x = H_alfa, color = 'r')
        plt.axvline(x = H_beta, color = 'c')
        plt.axvline(x = H_gamma, color = 'darkorchid')
        #labels, text...
        plt.xlabel("Wavelength [nm]", fontsize=18)
        plt.ylabel("Flux [W nm$^{-1}$ m$^{-2}$]", fontsize=18)
        plt.title(ide, fontsize=18)
        ax = plt.gca()
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.text(660, 0.95, "H$\\alpha$", transform = trans, fontdict={'fontsize':14})
        plt.text(490, 0.95, "H$\\beta$", transform = trans, fontdict={'fontsize':14})
        plt.text(438, 0.95, "H$\\gamma$", transform = trans, fontdict={'fontsize':14})
        
        if plot==True:
            plt.show()
        
        if out_path is not None:
            plt.savefig(f'{out_path}/{ide}.png', bbox_inches = "tight", format = "png")
        
        plt.close()