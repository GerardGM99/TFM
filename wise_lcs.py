from __future__ import print_function
import numpy as np
import datetime
import glob
import os, sys
from astropy.io import fits
import subprocess
import matplotlib.lines as mlines
import matplotlib
from matplotlib import pylab as plt
import phot_utils
import fitsmanip
import json

import astropy
from astropy.time import Time
from astropy.table import Table, Column, vstack


def bin_lc(out_, binned=5, mydtype=None, separate_instrument=False):
    
    #out_.convert_bytestring_to_unicode()
    if mydtype is None:
        mydtype=[("inst","|S15"), ("filter","|S15"),  \
        ("mjd","<f4"),("mjderr","<f4"), ("mag","<f4"), ("magerr","<f4"), ("ATel","<f4"), \
        ("limit", "i1")]
        
    #outall.convert_bytestring_to_unicode()
    
    def bin_by_filter(out_):
        
        outall = Table(np.zeros([0], dtype=mydtype))
        #We bin for each filter
        for band in set(out_['filter']):
            out = out_[out_['filter']==band]
            
            #We bin the limits together or non-limits together.
            outlim = out[(out['limit']==True) | (out['limit']==1)]
            out = out[(out['limit']==False) | (out['limit']==0)]

            #Compute the unique MJD to be used as the centre of each bin
            mjdbin = set( np.round(out["mjd"]/binned)*binned)
            mjdbinlim = set( np.round(outlim["mjd"]/binned)*binned)
            
            #Create a new table containing the binned magnitudes. 
            #It will have the length of the unique mjd vector.
            outbin = Table(np.zeros([len(mjdbin)], dtype=mydtype))

            outbinlim = Table(np.zeros([len(mjdbinlim)], dtype=mydtype))

           
            #For each distinct mjd in the bin, we compute the average mangitudes,
            #and store it in the position "i" of the vector
            for i, m in enumerate(mjdbin):
                outi = out[np.round(out["mjd"]/binned)*binned==m]
                outbin[i]["inst"] = outi["inst"][0]
                outbin[i]["mjd"] = np.average(outi['mjd'])
                outbin[i]["mjderr"] = np.std(outi["mjd"])
                outbin[i]["mag"] = -2.5 * np.log10(np.average(10**(-0.4*outi['mag']), weights=1./outi['magerr'] )) #np.average(outi['mag'], weights=1./outi['magerr'])
                '''fluxdiff =  (10**(-0.4*outi['mag'])) - 10**(-0.4*(outi['mag']+outi['magerr']))
                err = np.sqrt(np.sum(fluxdiff**2))
                fluxwitherr = 10**(-0.4*outbin[i]['mag']) - err
                magwitherror = -2.5 * np.log10( (10**(-0.4*outbin[i]['mag'])) - fluxwitherr)
                
                #outbin[i]["magerr"] = -2.5 * np.log10 (10**(-0.4*outi['mag'])) 
                outbin[i]["magerr"] = outbin[i]["mag"] - magwitherror'''
                outbin[i]["magerr"] = np.median(outi["magerr"])
                outbin[i]["filter"] = band
                outbin[i]["limit"] = False

            #For each distinct mjd in the bin, we compute the average mangitudes,
            #and store it in the position "i" of the vector
            for i, m in enumerate(mjdbinlim):
                outi = outlim[np.round(outlim["mjd"]/binned)*binned==m]
                outbinlim[i]["inst"] = outi["inst"][0]
                outbinlim[i]["mjd"] = np.average(outi['mjd'])
                outbinlim[i]["mjderr"] = np.std(outi["mjd"])
                outbinlim[i]["mag"] = -2.5 * np.log10(np.average(10**(-0.4*outi['mag']) )) #np.average(outi['mag'], weights=1./outi['magerr'])
                outbinlim[i]["magerr"] = -1
                outbinlim[i]["filter"] = band
                outbinlim[i]["limit"] = True
                
            #We append the binned magnitudes for that instrument to the general out array.
            #outall = np.append(outall, outbin, axis=0)
            outall = vstack([outall, outbin])


            #We append the limits
            #outall = np.append(outall, outbinlim, axis=0)
            outall = vstack([outall, outbinlim])
    
        return outall

    if separate_instrument:   
    
        outall = Table(np.zeros([0], dtype=mydtype))

        #We bin for each distinct instrument.
        for inst in set(out_['inst']):
            outinst = out_[out_['inst']==inst]
        outall = vstack([outall, outinst])
    else:
        outall = bin_by_filter(out_)
        
    return outall
        

def get_neowise_lightcurve(neowisefile, wisefile=None, composite=None, ax=None, ref_mjd=0, binsize=50, plot=False, limits=False, sigclip=True):
    '''
    neowisefile: NEOWISE-R Single Exposure (L1b) Source Table
    wisefile: AllWISE Multiepoch Photometry Table
    
    '''
    if ax is None and plot:
        plt.figure(figsize=(6,4))
        ax = plt.gca()

    if composite:
        t = Table.read(composite, format='ascii.ipac')
        w1 = t['w1mpro'][0]
        w1err = t['w1sigmpro'][0]
        w2 = t['w2mpro'][0]
        w2err = t['w2sigmpro'][0]
    

    #Reading NEOWISE
    if (neowisefile is not None):

        t = Table.read(neowisefile, format='ascii.ipac')
        t['mjd'] = t['mjd'] - ref_mjd
        #Remove upper limits
        if limits:
            mask = np.repeat(False, len(t))
        else:
            mask = ~np.array([('U' in s) for s in t['ph_qual']])
            #"ph_qual" values of "A," "B," or "C,"
            #mask = np.array([('A' in s) for s in t['ph_qual']]) | np.array([('B' in s) for s in t['ph_qual']]) | np.array([('C' in s) for s in t['ph_qual']])
            mask = mask * np.array( (t['w1snr']>2) * (t['w2snr']>2) * (t['w1rchi2']<150) * (t['w2rchi2']<150) )
            
        if type(t['w2mpro']) == astropy.table.column.MaskedColumn:
            mask2 = ~t['w2mpro'].mask
        else:
            mask2 = np.repeat(True, len (t))
            
            
        if sigclip:
            med1 = np.median(t['w1mpro'][mask])
            med2 = np.median(t['w2mpro'][mask*mask2])
            
            std1 = np.std(t['w1mpro'][mask])
            std2 = np.std(t['w1mpro'][mask*mask2])
            
            sigma1 =  np.abs(t['w1mpro'][mask] - med1) < 2*std1
            sigma2 =  np.abs(t['w2mpro'][mask*mask2] - med2) < 2*std2
        else:
            sigma1 = np.repeat(True, len(t[mask]))
            sigma2 = np.repeat(True, len(t[mask*mask2]))



        #mask = np.array([('U' in s) or ('C' in s) for s in t['ph_qual']])
        
        #Bin the lightcurve using the errors
        if binsize > 0:
            x1, y1, xerr1, yerr1 =  phot_utils.bin_lightcurve(t['mjd'][mask][sigma1], t['w1mpro'][mask][sigma1], \
                                                              yerr=t['w1sigmpro'][mask][sigma1], binsize=binsize, mode="average")
            x2, y2, xerr2, yerr2 =  phot_utils.bin_lightcurve(t['mjd'][mask*mask2][sigma2], t['w2mpro'][mask*mask2][sigma2], \
                                                              yerr=t['w2sigmpro'][mask*mask2][sigma2], binsize=binsize, mode="average")
        else:
            x1 = t['mjd'][mask][sigma1]
            xerr1 = np.zeros_like(x1)
            y1 = t['w1mpro'][mask][sigma1]
            yerr1 = t['w1sigmpro'][mask][sigma1]

            x2 = t['mjd'][mask][sigma2]
            xerr2 = np.zeros_like(x2)
            y2 = t['w2mpro'][mask][sigma2]
            yerr2 = t['w2sigmpro'][mask][sigma2]
            
        out = np.zeros(len(x1) + len(x2), dtype=[("inst","|S15"), ("filter","|S15"),  ("mjd","<f8"), \
                        ("mjderr","<f8"), ("mag","<f8"), ("magerr","<f8"), ("ATel","<i8"), ("limit","i1")])
        out['inst'] = 'NEOWISE'
        out['mjd'][0:len(x1)] = x1
        out['mjderr'][0:len(x1)] = xerr1
        out['mag'][0:len(x1)] = y1
        out['magerr'][0:len(x1)] = yerr1
        
        out['filter'][0:len(x1)] = 'W1'
        out['mjd'][len(x1):] = x2
        out['mjderr'][len(x1):] = xerr2
        out['mag'][len(x1):] = y2
        out['magerr'][len(x1):] = yerr2
        out['filter'][len(x1):] = 'W2'
        
        if plot:
            ax.errorbar(x1, y1, yerr=yerr1, xerr=xerr1, fmt="o", label="W1", elinewidth=2, mfc="blue", mec="blue", ecolor="blue")
            ax.errorbar(x2, y2, yerr=yerr2, xerr=xerr2, fmt="o", label="W2", elinewidth=2, mfc="orange", mec="orange", ecolor="orange")            
    else:    
        out = np.zeros(0, dtype=[("inst","|S15"), ("filter","|S15"), ("mjd","<f6"), \
                       ("mjderr","<f6"), ("mag","<f6"), ("magerr","<f6"), ("ATel","<i6"), ("limit","i1")])
    
    out = Table(out)

    #Reading WISE
    if (wisefile is None):
        t_joined = out
    #Reading WISE
    if (wisefile is not None):
        t = Table.read(wisefile, format='ascii.ipac')
        t['mjd'] = t['mjd'] - ref_mjd

       
        #Bin the lightcurve using the errors
        try:
            mask3 = ~t['w3mpro_ep'].mask
            mask4 = ~t['w4mpro_ep'].mask
        except AttributeError:
            mask3 = np.array([f[0] =='0' for f in t['cc_flags']])
            mask4 = np.array([f[1]=='0' for f in t['cc_flags']])
            
        if sigclip:
            med1 = np.nanmedian(t['w1mpro_ep'])
            med2 = np.nanmedian(t['w2mpro_ep'])
            med3 = np.nanmedian(t['w3mpro_ep'].data[mask3])
            med4 = np.nanmedian(t['w4mpro_ep'].data[mask4])
            
            std1 = np.nanstd(t['w1mpro_ep'])
            std2 = np.nanstd(t['w2mpro_ep'])
            std3 = np.nanstd(t['w3mpro_ep'][mask3])
            std4 = np.nanstd(t['w4mpro_ep'][mask4])
            
            sigma1 =  np.abs(t['w1mpro_ep'] - med1) < 2*std1
            sigma2 =  np.abs(t['w2mpro_ep'] - med2) < 2*std2
            sigma3 =  np.abs(t['w3mpro_ep'] - med3) < 2*std3
            sigma4 =  np.abs(t['w4mpro_ep'] - med4) < 2*std4
            
            mask1 = sigma1
            mask2 = sigma2
            mask3 = sigma3
            mask4 = sigma4
            
            
        x1, y1, xerr1, yerr1 =  phot_utils.bin_lightcurve(t['mjd'][mask1], t['w1mpro_ep'][mask1], yerr=t['w1sigmpro_ep'][mask1], binsize=binsize, mode="average")
        x2, y2, xerr2, yerr2 =  phot_utils.bin_lightcurve(t['mjd'][mask2], t['w2mpro_ep'][mask2], yerr=t['w2sigmpro_ep'][mask2], binsize=binsize, mode="average")
        x3, y3, xerr3, yerr3 =  phot_utils.bin_lightcurve(t['mjd'][mask3], t['w3mpro_ep'][mask3], yerr=t['w3sigmpro_ep'][mask3], binsize=binsize, mode="average")
        x4, y4, xerr4, yerr4 =  phot_utils.bin_lightcurve(t['mjd'][mask4], t['w4mpro_ep'][mask4], yerr=t['w4sigmpro_ep'][mask4], binsize=binsize, mode="average")
            
        N1 = len(x1)
        N2 = N1 + len(x2)
        N3 = N2 + len(x3)
        N4 = N3 + len(x4)
        
        out2 = np.zeros(N4, dtype=[("inst","|S15"), ("filter","|S15"),  ("mjd","<f4"), \
                        ("mjderr","<f4"), ("mag","<f4"), ("magerr","<f4"), ("ATel","<f4"), ("limit","i1")])

        out2['inst'] = 'WISE'
        out2['mjd'][0:N1] = x1
        out2['mjderr'][0:N1] = xerr1
        out2['mag'][0:N1] = y1
        out2['magerr'][0:N1] = yerr1
        out2['filter'][0:N1] = 'W1'
        
        out2['mjd'][N1:N2] = x2
        out2['mjderr'][N1:N2] = xerr2
        out2['mag'][N1:N2] = y2
        out2['magerr'][N1:N2] = yerr2
        out2['filter'][N1:N2] = 'W2'

        out2['mjd'][N2:N3] = x3
        out2['mjderr'][N2:N3] = xerr3
        out2['mag'][N2:N3] = y3
        out2['magerr'][N2:N3] = yerr3
        out2['filter'][N2:N3] = 'W3'

        out2['mjd'][N3:] = x4
        out2['mjderr'][N3:] = xerr4
        out2['mag'][N3:] = y4
        out2['magerr'][N3:] = yerr4
        out2['filter'][N3:] = 'W4'
        
        out2 = Table(out2)

        t_joined = vstack([out, out2])
        
        if plot:
            ax.errorbar(x1, y1, yerr=yerr1, xerr=xerr1, fmt="o", elinewidth=2, mfc="blue", mec="blue", ecolor="blue")
            ax.errorbar(x2+2, y2, yerr=yerr2, xerr=xerr2, fmt="o", elinewidth=2, mfc="orange", mec="orange", ecolor="orange")
            ax.errorbar(x3+3, y3, yerr=yerr3, xerr=xerr3, fmt="o", elinewidth=2, mfc="red", mec="red", ecolor="red", label="W3")
            ax.errorbar(x4+4, y4, yerr=yerr4, xerr=xerr4, fmt="o", elinewidth=2, mfc="brown", mec="brown", ecolor="brown", label="W4")
      

    if composite:
        xmin, xmax = ax.get_xlim()
        ax.fill_between( np.array([xmin, xmax]), np.array(2*[w1-w1err]), np.array(2*[w1+w1err]), color="blue", alpha=0.1)
        ax.fill_between([xmin, xmax], np.array(2*[w2-w2err]), np.array(2*[w2+w2err]), color="orange", alpha=0.1)
        ax.hlines(w1, xmin, xmax, color="blue", linestyle="dotted")
        ax.hlines(w2, xmin, xmax, color="orange", linestyle="dotted")
        plt.xlim(xmin, xmax)
        
    if plot:
        ax.invert_yaxis()
        if ref_mjd ==0:
            ax.set_xlabel("MJD")
        else:
            ax.set_xlabel("MJD - %.1f [days]"%ref_mjd)
        ax.set_ylabel("Magnitude")
        ax.legend(loc="best")
        #ax.legend(loc="lower right")
    
        plt.minorticks_on()    
        plt.tight_layout()   
        #plt.savefig(os.path.join(plotdir, "wise_lightcurve.pdf"), ddp=200)
        
    t_joined.sort(keys="mjd")
    return t_joined


