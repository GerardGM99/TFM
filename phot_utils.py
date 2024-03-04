# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:11:26 2024

@author: xGeeRe
"""
import numpy as np
from uncertainties import ufloat

def bin_lightcurve(x, y, yerr=None, mode="median", binsize=3):
    '''
    Bins the lightcurve in bins of binsize.
    '''

    if binsize < 0:
        return x, y, yerr
    
    if yerr is None:
        yerr = np.zeros_like(y)
    
    if np.ndim(x) == 0:
        x = np.array([x])
        y = np.array([y])
        yerr = np.array([yerr])
    elif len(x) == 0:
        return np.array([]), np.array([]), np.array([])
        
        
    x_i = list(set(np.round(x/binsize) * binsize))
    x_i.sort()
    x_i = np.array(x_i)
    y_o = []
    yerr_o = []
    x_o = []
    xerr_o  = []
    

    for xi in x_i:
        mask = np.abs(x-xi) < binsize/2.
        mask = mask * ~(np.isnan(yerr))
        #If we only have one register, no binning needed
        if len(x[mask])==1:
            x_o.append(x[mask][0])
            y_o.append(y[mask][0])
            xerr_o.append(0)

            if not yerr is None:
                yerr_o.append(yerr[mask][0])
            else:
                yerr_o.append([0])
        elif np.any(mask):
            #print ("OBJECTS PER EPOCH", len(y[mask]))
           
            x_o.append(np.average(x[mask]))
            xerr_o.append(np.std(x[mask]))
            #print(np.std(x[mask]))
            if mode == "median":
                y_o.append(np.median(y[mask]))
                if not yerr is None:
                    yerr_o.append(np.sqrt( np.sum( (yerr[mask])**2 )))
            else:
                fluxes = 10**(-0.4*y[mask])
                fluxerr = fluxes - 10**(-0.4*(y[mask] + yerr[mask]))
                fluxarr = np.array([ufloat(z) for z in zip(fluxes, fluxerr)])
                               
                #Weighted mean:
                #https://ned.ipac.caltech.edu/level5/Leo/Stats4_5.html
                avgflux = np.nansum(fluxes/fluxerr**2) / np.nansum(1./fluxerr**2)
                
                #avgmag = -2.5 * np.log10(fluxarr.mean().n)
                avgmag = -2.5 * np.log10(avgflux)
                
                #stdflux = np.sqrt(1./ np.nansum(1./fluxerr**2))
                #stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + stdflux))

                stdev_fl = np.std(fluxes[~np.isnan(fluxes)])
                stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + stdev_fl))
                
                #print (yerr[mask], len(fluxerr), stdev)
                
                #stdev = np.abs(-2.5 * np.log10(fluxarr.mean().n) +2.5 * np.log10(fluxarr.mean().n + fluxarr.mean().s))
                
                y_o.append(avgmag)
                yerr_o.append(stdev)

    x_o = np.array(x_o)
    xerr_o = np.array(xerr_o)

    y_o = np.array(y_o)
    yerr_o = np.array(yerr_o)
    
    return x_o, y_o, xerr_o, yerr_o  