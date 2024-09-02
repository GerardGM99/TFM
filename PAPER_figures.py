# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:17:10 2024

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
from astropy.io import fits
from extinction import fitzpatrick99, remove, calzetti00
from phot_utils import density_scatter
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.ticker as ticker
from specutils import Spectrum1D
from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
from astropy.stats import sigma_clip
import shutil

plt.rcParams['ytick.direction']='in'
plt.rcParams['xtick.direction']='in'
plt.rcParams['axes.axisbelow'] = False
# plt.rcParams['font.size']= 10
plt.rcParams['font.family']= 'serif'

def axes_params(ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, legend=False,
                title_size=14, label_size=14):
    ax.set_title(title, fontsize=title_size, weight='bold')
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    
    if legend:
        ax.legend()
        
def original_lc(file_name, ref_mjd=58190.45, y_ax='mag', outliers='median', n_std=3, binsize=0, 
                rang=None, plot=False, savefig=False, ax=None, add_secondary_xaxis=True):

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
                
                sigma = remove_outliers(file['mag'][file['filter'] == band], method=outliers)
                    
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
                else:
                    ax.errorbar(t_observed[sigma] - ref_mjd, y_observed[sigma], xerr=t_err, yerr=uncert[sigma], color=plot_color, label=band, 
                                fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                ax.set_ylabel("Magnitude", family="serif", fontsize=15)
                    
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
                ax.set_ylabel("Flux", family="serif", fontsize=15)
                    
        ax.set_title(f"{source_name}", weight="bold") 
        ax.set_xlabel(f"MJD - {ref_mjd} [days]", family="serif", fontsize=15)
        ax.tick_params(which='major', width=2, direction='out')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, direction='out')
        ax.tick_params(labelsize=15)
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
            ay2.tick_params(labelsize=15)
            ay2.set_xlim([xmin2, xmax2])
            ay2.set_xlabel('Year', fontsize=15)
            ay2.ticklabel_format(useOffset=False)
            
        colors_for_filters = [color_dict.get((ins, filt), None) for filt in available_filters]
        
        if 'ZTF' in ins:
            loc='upper left'
        else:
            loc='upper right'
        ax.legend(loc=loc, ncols=len(available_filters), labelcolor=colors_for_filters, 
                  shadow=True, columnspacing=0.8, handletextpad=0.5, handlelength=1, markerscale=1.5, 
                  prop={"weight": "bold", 'size':12})
        
        if savefig:
            plt.savefig(f'{out_path}/{source_name}.png', bbox_inches="tight", format="png")
        
        if plot:
            plt.show()
        
        if ax is None:
            plt.close()

def stacked_og_lc(name, ins):
    
    # Create a new figure with GridSpec
    fig = plt.figure(constrained_layout=True, figsize=(6.5,2.5*len(ins)))
    gs = gridspec.GridSpec(len(ins), 1, figure=fig, hspace=0)  # hspace=0 to remove separation between plots
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    axes=[ax1, ax2, ax3, ax4]
    
    # Plot the light curves on the respective axes
    for i, instrument in enumerate(zip(ins,axes)):
        if i == 0:
            secondary = True
        else:
            secondary = False
        
        if instrument[0]=='NEOWISE':
            binsize=50
        else:
            binsize=0
        original_lc(f'{instrument[0]}_lightcurves_std/{name}.csv', ax=instrument[1], plot=False, 
                    savefig=False, add_secondary_xaxis=secondary, binsize=binsize)
    
    for j, ax in enumerate(axes):
        ax.set_title('')
        if j==1 or j==0 or j==2:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False, direction='in')
            ax.tick_params(which='minor', length=4, direction='in')
        
    
    plt.tight_layout()
    # plt.savefig(f'cool_plots/stacked_lcs/{name}.png', bbox_inches="tight", format="png")
    # plt.close()
    
def show_lines(ax, lines_file, xrange, priority):

    if lines_file is not None:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        lines = Table.read(lines_file, format='ascii')
        lines_sorted = lines[np.argsort(lines['wavelength'])]
        prev_line_wl = None
        prev_line_wl2 = None
        prev_ymax = 0.83+0.13
        # prev_ymax2 = 0.75
        delta_perce2 = 1
        for line, wl, prio, color in zip(lines_sorted['line'], lines_sorted['wavelength'], lines_sorted['prio'], lines_sorted['color']):
            if (xrange[0] <= wl <= xrange[1]) & (prio in priority):
                # color = cmap(float(np.where(unique_priorities == prio)[0]) / (num_priorities - 1))
                if prev_line_wl is not None:
                    delta_wl = wl - prev_line_wl
                    delta_perce = delta_wl / (xrange[1] - xrange[0])
                    if prev_line_wl2 is not None:
                        delta_wl2 = wl - prev_line_wl2
                        delta_perce2 = delta_wl2 / (xrange[1] - xrange[0])
                    if (delta_perce < 0.05) & (prev_ymax == 0.83+0.13) & (delta_perce2 > 0.02):
                        ymax = 0.80+0.13
                    elif (delta_perce < 0.05) & (prev_ymax == 0.83+0.13) & (delta_perce2 < 0.02):
                        ymax = 0.77+0.13
                    else:
                        ymax = 0.83+0.13
                else:
                    ymax=0.83+0.13
                ax.axvline(wl, ymax=ymax, ls='dotted', alpha=0.5, color=color)
                ax.text(wl, ymax+0.01, line, transform = trans, fontdict={'fontsize':12}, rotation = 90, ha='center')
                prev_line_wl2 = prev_line_wl
                prev_line_wl = wl
                # prev_ymax2 = prev_ymax
                prev_ymax = ymax

def all_low_spectra():
    directory = 'data/all_spectra'
    files = os.listdir(directory)
    
    fig = plt.figure(constrained_layout=True, figsize=(12, 14))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.01, width_ratios=[1, 0.2])  # hspace=0 to remove separation between plots
    gs01 = gs[1].subgridspec(13, 1, hspace=0)
    
    ax = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs01[0,0])
    ax2 = fig.add_subplot(gs01[1,0])
    ax3 = fig.add_subplot(gs01[2,0])
    ax4 = fig.add_subplot(gs01[3,0])
    ax5 = fig.add_subplot(gs01[4,0])
    ax6 = fig.add_subplot(gs01[5,0])
    ax7 = fig.add_subplot(gs01[6,0])
    ax8 = fig.add_subplot(gs01[7,0])
    ax9 = fig.add_subplot(gs01[8,0])
    ax10 = fig.add_subplot(gs01[9,0])
    ax11 = fig.add_subplot(gs01[10,0])
    ax12 = fig.add_subplot(gs01[11,0])
    ax13 = fig.add_subplot(gs01[12,0])
    
    axes = [ax13, ax12, ax11, ax10, ax9, ax8, ax7, ax6, ax5, ax4, ax3, ax2]
    #fig, ax = plt.subplots(figsize=(12, 14))
    
    show_lines(ax, 'data/spectral_lines.txt', xrange=[3900, 9000], priority=['1'])
    ax.axvspan(6870, 6950, alpha=0.3, color='lightsteelblue')
    ax.axvspan(7590, 7700, alpha=0.3, color='lightsteelblue')
    ax.axvspan(7150, 7350, alpha=0.3, color='lightsteelblue')
    
    constant = 0
    triger = False
    i = 0
    for file in files:
        if file.endswith('.txt'):
            input_filename=f'{directory}/{file}'
            
            if file.startswith('spectra1D'):
                asciicords='data/TFM_table.csv'
                fits_file = input_filename.split(".")[0]+'.fits'
                hdu = fits.open(fits_file)
                ra = hdu[0].header['RA']
                dec = hdu[0].header['DEC']
                hdu.close()
                
                # Xmatch with the asciicoords file
                table = Table({'ra':[ra], 'dec':[dec]})
                table_coord = Table.read(asciicords, format='ascii.csv')
                column_names = ['ra', 'dec', 'RA', 'DEC', 'Gaia source ID']
                xmatch_table = cu.sky_xmatch(table, table_coord, 1800, column_names)
                source_name = xmatch_table['Name'][0]
            else:
                asciicords='data/TFM_table.csv'
                table_coord = Table.read(asciicords, format='ascii.csv')
                source_name = file.split('.')[0]
                triger = True

    
            data = Table.read(input_filename, format='ascii')
                
            Av = table_coord['$A_V$'][table_coord['Gaia source ID']==int(source_name)][0]
            #'deredden' flux using Fitzpatrick (1999)
            flux = remove(fitzpatrick99(np.array(data['wavelength']), Av, 3.1), np.array(data['flux']))
            mean_flux = np.mean(flux)
            if triger is True:
                wl = data['wavelength'][(flux>=0)&(data['wavelength']>3800)]
                flu = constant+flux[(flux>=0)&(data['wavelength']>3800)]/mean_flux
                clipped_flux = sigma_clip(flu, sigma=10)
                ax.plot(wl, clipped_flux, color='grey', alpha=0.4)
                spec1 = Spectrum1D(spectral_axis=wl * u.AA, flux=clipped_flux * u.m / u.m)
                spec1_gsmooth = gaussian_smooth(spec1, stddev=10)
                clipped_flux = sigma_clip(spec1_gsmooth.flux, sigma = 5)
                ax.plot(spec1_gsmooth.spectral_axis, clipped_flux, color='k')
            else:
                wl = data['wavelength'][(flux>=0)&(data['wavelength']>3800)]
                flu = constant+flux[(flux>=0)&(data['wavelength']>3800)]/mean_flux
                ax.plot(wl, flu, color='k')
                axes[i].plot(wl, flu/np.mean(flu[(wl>6554)*(wl<6575)]), color='k')
            
            if source_name=='2164630463117114496' or source_name=='1870955515858422656' or source_name=='4299904519833646080':
                ax.text(7200, mean_flux+constant+1.5, 'Gaia '+source_name, fontsize=15)
            elif source_name=='2030965725082217088':
                ax.text(7200, mean_flux+constant+0.8, 'Gaia '+source_name, fontsize=15)
            elif source_name=='4076568861833452160':
                ax.text(7200, mean_flux+constant+1.4, 'Gaia '+source_name, fontsize=15)
            else:
                ax.text(7200, mean_flux+constant+1, 'Gaia '+source_name, fontsize=15)
            constant += 4
            triger = False
            i += 1
    
    x3, y3 = su.lamost_spectra(f'{directory}/spec-56948-HD052351N362354B_sp14-213.fits', 'data/TFM_table.csv', dered='fitz',
                               plot=False, ax=None, color='k')
    
    mean_flux = np.mean(y3)
    ax.plot(x3, constant+y3/mean_flux, color='k')
    ax.text(7200, 49, 'Gaia '+'187219239343050880', fontsize=15)
    # constant += 4
    
    # hdu = fits.open('data/all_spectra/med-58149-HIP265740401_sp09-191.fits')
    # ra = hdu[0].header['RA']
    # dec = hdu[0].header['DEC']      
    # table = Table({'ra':[ra], 'dec':[dec]})
    # table_coord = Table.read(asciicords, format='ascii.csv')
    # column_names = ['ra', 'dec', 'RA', 'DEC', 'Gaia source ID']
    # xmatch_table = cu.sky_xmatch(table, table_coord, 1800, column_names)
    # source_name = xmatch_table['Name'][0]
        
    # for i in range(1, len(hdu)):
    #     data = hdu[i].data
    #     hdr = hdu[i].header
    #     band = hdr['EXTNAME']
        
    #     if band.startswith('COADD'):
    #         ax.plot(data['WAVELENGTH'][0], constant+data['FLUX'][0]/np.mean(data['FLUX'][0]), color='k')
      
    ax.set_xlabel(r'Rest Wavelength [$\AA$]', fontsize=15)
    ax.set_ylabel(r'Normalized Flux + constant', fontsize=15)
    ax.set_ylim(bottom=-1, top=55.5)
    ax.set_xlim(left=3750, right=8999)
    ax.tick_params(axis='both', labelsize=15)
    
    for axi in axes:
        axi.set_xlim(6554, 6575)
        axi.set_ylim(0.95, 1.05)
        axi.axvline(6562.8, ls='dashed', color='r')
        # ax1.text(6563.5, 1.12, "H$\\alpha$", fontdict={'fontsize': 15})
        axi.axes.get_yaxis().set_visible(False)
        axi.axes.get_xaxis().set_visible(False)
        
    ax1.set_xlim(6554, 6575)
    ax1.set_ylim(0.9, 1.1)
    ax1.axvline(6562.8, ls='dashed', color='r')
    # ax1.text(6563.5, 1.12, "H$\\alpha$", fontdict={'fontsize': 15})
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax13.axes.get_xaxis().set_visible(True)
            
def lamost_lowres():
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    show_lines(ax, 'data/spectral_lines.txt', xrange=[3900, 9000], priority=['1'])
    
    directory = 'data/lamost_spectra'
    x1, y1 = su.lamost_spectra(f'{directory}/spec-55977-GAC_080N37_V1_sp03-241.fits', 'data/TFM_table.csv', 
                               dered='fitz', plot=False, ax=ax)
    x2, y2 = su.lamost_spectra(f'{directory}/spec-56687-GAC080N37V1_sp03-241.fits', 'data/TFM_table.csv', dered='fitz',
                               plot=False, ax=ax, color='blue')
    x3, y3 = su.lamost_spectra(f'{directory}/spec-56948-HD052351N362354B_sp14-213.fits', 'data/TFM_table.csv', dered='fitz',
                               plot=False, ax=ax, color='orange')
    
    inset_ax = inset_axes(ax, width="100%", height="100%", 
                  bbox_to_anchor=(7000, 9000, 900, 6000), 
                  bbox_transform=ax.transData, 
                  loc='upper left')
    inset_ax.plot(x1, y1/np.mean(y1[(x1>6554)*(x1<6575)]), color='k')
    inset_ax.plot(x2, y2/np.mean(y2[(x2>6554)*(x2<6575)]), color='blue')
    inset_ax.plot(x3, y3/np.mean(y3[(x3>6554)*(x3<6575)]), color='orange')
    inset_ax.set_xlim(6554, 6575)
    inset_ax.set_ylim(0.9, 1.15)
    inset_ax.axvline(6562.8, ls='dashed', color='r')
    inset_ax.text(6563.5, 1.12, "H$\\alpha$", fontdict={'fontsize': 15})
    inset_ax.set_ylabel('Normalized flux', fontsize=15)
    inset_ax.set_title('Zoom around H$\\alpha$', fontsize=15)
    
    ax.text(7000, 7000, '19-02-2012', fontsize=15, color='k')
    ax.text(7000, 4000, '29-01-2014', fontsize=15, color='blue')
    ax.text(7000, 1100, '17-10-2014', fontsize=15, color='orange')        
    ax.set_title('Gaia 187219239343050880', fontsize=16, weight='bold')
    ax.set_ylim(top=18000, bottom=0)
    ax.set_xlim(left=3600, right=9100)

def hermes():
    
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(7, 10))
    
    show_lines(ax1, 'data/spectral_lines.txt', xrange=[6558,6570], priority=['1'])
    show_lines(ax2, 'data/spectral_lines.txt', xrange=[5887,5898], priority=['1','3'])
    
    name='6123873398383875456'
    directory = 'data/HERMES_spectra'
    spec = pd.read_csv(f'{directory}/{name}_11-5.txt', sep=' ')
    su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[6558,6570], #Av=5.7289376,
                units=['$\AA$','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'],
                plot=False, ax=ax1, color='k')
    spec = pd.read_csv(f'{directory}/{name}_13-5.txt', sep=' ')
    su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[6558,6570], #Av=5.7289376,
                units=['$\AA$','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'],
                plot=False, ax=ax1, color='blue', alpha=0.5)
    
    ax1.text(6558.5, 7e-16, '11-05-2024', fontsize=15, color='k')
    ax1.text(6558.5, 6.5e-16, '13-05-2024', fontsize=15, color='blue', alpha=0.5)
    
    spec = pd.read_csv(f'{directory}/{name}_11-5.txt', sep=' ')
    su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[5885,5898], #Av=5.7289376,
                units=['$\AA$','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'],
                plot=False, ax=ax2, color='k')
    spec = pd.read_csv(f'{directory}/{name}_13-5.txt', sep=' ')
    su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=[5885,5898], #Av=5.7289376,
                units=['$\AA$','$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$'],
                plot=False, ax=ax2, color='blue', alpha=0.5)
    
    # ax.text(6530, 4.5e-16, '11-05-2024', fontsize=15, color='k')
    # ax.text(6530, 4e-16, '13-05-2012', fontsize=15, color='blue')      
    ax1.set_title('Gaia 6123873398383875456', fontsize=16, weight='bold')

def H_velocity_spec():
    spec_dict = {'2030965725082217088':'0200', '2074693061975359232':'0201',
                 '2164630463117114496':'0202', '2206767944881247744':'0203',
                 '1870955515858422656':'0204', '2013187240507011456':'0205',
                 '2060841448854265216':'0206', '2166378312964576256':'0207'}
    fies = ['4076568861833452160', '4299904519833646080', '2200433413577635840']
    lamost = {'3444168325163139840':'med-58149-HIP265740401_sp09-191.fits'}
    hermes = ['6123873398383875456']
    
    # Create a new figure with GridSpec
    fig = plt.figure(constrained_layout=True, figsize=(12, 3*3))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)  # hspace=0 to remove separation between plots
    gs01 = gs[0].subgridspec(4, 2, hspace=0.05, wspace=0.05)
    gs02 = gs[1].subgridspec(4, 2, hspace=0.05, wspace=0.05)
    
    ax0 = fig.add_subplot(gs01[0,0])
    ax1 = fig.add_subplot(gs01[1,0], sharex=ax0)
    ax2 = fig.add_subplot(gs01[2,0], sharex=ax0)
    ax3 = fig.add_subplot(gs01[3,0], sharex=ax0)
    ax4 = fig.add_subplot(gs01[0,1], sharey=ax0)
    ax5 = fig.add_subplot(gs01[1,1], sharex=ax4, sharey=ax1)
    ax6 = fig.add_subplot(gs01[2,1], sharex=ax4, sharey=ax2)
    ax7 = fig.add_subplot(gs01[3,1], sharex=ax4, sharey=ax3)
    axesL=[ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    ax8 = fig.add_subplot(gs02[0,0])
    ax9 = fig.add_subplot(gs02[1,0], sharex=ax8)
    ax10 = fig.add_subplot(gs02[2,0], sharex=ax8)
    ax13 = fig.add_subplot(gs02[3,0], sharex=ax8)
    ax11 = fig.add_subplot(gs02[0,1])
    ax12 = fig.add_subplot(gs02[1,1], sharex=ax11)
    ax14 = fig.add_subplot(gs02[2,1], sharex=ax11)
    ax15 = fig.add_subplot(gs02[3,1], sharex=ax11)
    axes_fies=[ax8, ax9, ax10]
    axes_lamostL=[ax12, ax14, ax15]
    
    
    #CAFOS
    for ax, name in zip(axesL,spec_dict):
        unit = spec_dict.get(name)
        spectrum = Table.read(f'data/cafos_spectra/spectra1D_dswfz_uniB_{unit}.txt', format='ascii')
        mask = (spectrum['wavelength']>6500)*(spectrum['wavelength']<6620)
        mask2 = (spectrum['wavelength']>4800)*(spectrum['wavelength']<4920)
        
        su.spec_velocity(6562.8, spectrum['wavelength'][mask], spectrum['flux'][mask], legend=False,
                         color='r', ax=ax)
        su.spec_velocity(4861, spectrum['wavelength'][mask2], spectrum['flux'][mask2],  legend=False,
                         color='b', ax=ax)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        ax.text(0.5, 0.95, name, ha='center', va='top', fontsize=11, transform=trans, weight='bold')
        ax.set_xlim(left=-2400, right=2400)
        ax.set_ylabel('')
        ax.set_xlabel('')
    
    for ax in axesL:
        if ax in [ax0, ax3]:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(top=ymax+0.1)
        elif ax in [ax1, ax2]:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(top=ymax+0.4)
            
        
    ax3.set_xticks([-1500,0,1500])
    # ax3.tick_params(axis='x', which='minor', length=5, width=1)
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    
    ax7.set_xticks([-1500,0,1500])
    # ax7.tick_params(axis='x', which='minor', length=5, width=1)
    ax7.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    
    #FIES
    for ax, name in zip(axes_fies,fies):
        spectrum = Table.read(f'data/FIES-M_spectra/{name}.txt', format='ascii')
        mask = (spectrum['wavelength']>6500)*(spectrum['wavelength']<6620)
        mask2 = (spectrum['wavelength']>4800)*(spectrum['wavelength']<4920)
        
        su.spec_velocity(6562.8, spectrum['wavelength'][mask], spectrum['flux'][mask], legend=False,
                         color='r', ax=ax)
        su.spec_velocity(4861, spectrum['wavelength'][mask2], spectrum['flux'][mask2],  legend=False,
                         color='b', ax=ax)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        ax.text(0.5, 0.95, name+'\nFIES-M', ha='center', va='top', fontsize=11, transform=trans, weight='bold')
        ax.set_xlim(left=-400, right=400)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
    ax8.set_ylim(top=1.8)
    ax9.set_ylim(top=1.8)
    ax10.set_ylim(top=1.8)
    
    #LAMOST-M
    hdu = fits.open('data/lamost_spectra/med-58149-HIP265740401_sp09-191.fits')
    
    data_R = hdu[2].data
    mask = (data_R['WAVELENGTH'][0]>6500)*(data_R['WAVELENGTH'][0]<6620)
    
    su.spec_velocity(6562.8, data_R['WAVELENGTH'][0][mask], data_R['FLUX'][0][mask], legend=False,
                     color='r', ax=ax11)
    trans = transforms.blended_transform_factory(ax11.transAxes, ax11.transAxes)
    ax11.text(0.5, 0.95, '3444168325163139840\nLAMOST-M', ha='center', va='top', fontsize=11, transform=trans, weight='bold')
    ax11.set_xlim(left=-700, right=700)
    ax11.set_ylim(bottom=0.8, top=1.8)
    ax11.set_ylabel('')
    ax11.set_xlabel('')
    
    #LAMOST-L
    files = os.listdir('data/lamost_spectra')
    flies = []
    for f in files:
        if os.path.basename(f).startswith('spec'):
            flies.append(f)
    
    flag = 1
    for input_file,ax in zip(flies,axes_lamostL):
        hdu = fits.open(f'data/lamost_spectra/{input_file}')
        
        data = hdu[1].data
        mask = (data['WAVELENGTH'][0]>6500)*(data['WAVELENGTH'][0]<6620)
        mask2 = (data['WAVELENGTH'][0]>4800)*(data['WAVELENGTH'][0]<4920)
        
        su.spec_velocity(6562.8, data['WAVELENGTH'][0][mask], data['FLUX'][0][mask], legend=False,
                         color='r', ax=ax)
        su.spec_velocity(4861, data['WAVELENGTH'][0][mask2], data['FLUX'][0][mask2],  legend=False,
                         color='b', ax=ax)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        if flag==1:
            ax.text(0.5, 0.95, '187219239343050880\nLAMOST-L', ha='center', va='top', fontsize=11, transform=trans, weight='bold')
        # ax.set_xlim(left=-700, right=700)
        ax.set_ylim(top=1.3)
        ax.set_ylabel('')
        ax.set_xlabel('')
        flag+=1
            
    
    #HERMES
    spectrum = Table.read('data/HERMES_spectra/6123873398383875456_11-5.txt', format='ascii')
    mask = (spectrum['wavelength']>6500)*(spectrum['wavelength']<6620)
    mask2 = (spectrum['wavelength']>4800)*(spectrum['wavelength']<4920)
    
    su.spec_velocity(6562.8, spectrum['wavelength'][mask], spectrum['flux'][mask], legend=False,
                     color='r', ax=ax13, label=r'H$\alpha$')
    su.spec_velocity(4861, spectrum['wavelength'][mask2], spectrum['flux'][mask2],  legend=False,
                     color='b', ax=ax13, label=r'H$\beta$')
    trans = transforms.blended_transform_factory(ax13.transAxes, ax13.transAxes)
    ax13.text(0.5, 0.95, '6123873398383875456\nHERMES', ha='center', va='top', fontsize=11, transform=trans, weight='bold')
    ax13.set_ylim(top=1.35)
    ax13.set_ylabel('')
    ax13.set_xlabel('')
    ax13.legend(loc='upper center', bbox_to_anchor=(1.55, -0.12), ncol=2, fontsize=13)
    
    fig.text(0.5, 0.07, 'Velocity [km/s]', ha='center', va='center', fontsize=15)
    fig.text(0.07, 0.5, 'Normalized Flux', ha='center', va='center', rotation='vertical', fontsize=15)
    
    ax0.tick_params(axis='x', labelbottom=False, direction='in')
    ax1.tick_params(axis='x', labelbottom=False, direction='in')
    ax2.tick_params(axis='x', labelbottom=False, direction='in')
    ax4.tick_params(axis='x', labelbottom=False, direction='in')
    ax5.tick_params(axis='x', labelbottom=False, direction='in')
    ax6.tick_params(axis='x', labelbottom=False, direction='in')
    ax8.tick_params(axis='x', labelbottom=False, direction='in')
    ax9.tick_params(axis='x', labelbottom=False, direction='in')
    ax10.tick_params(axis='x', labelbottom=False, direction='in')
    ax11.tick_params(axis='x', labelbottom=False, direction='in')
    ax12.tick_params(axis='x', labelbottom=False, direction='in')
    ax14.tick_params(axis='x', labelbottom=False, direction='in')
    
    ax4.tick_params(axis='y', labelleft=False, direction='in')
    ax5.tick_params(axis='y', labelleft=False, direction='in')
    ax6.tick_params(axis='y', labelleft=False, direction='in')
    ax7.tick_params(axis='y', labelleft=False, direction='in')
    ax11.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False, direction='in')
    ax12.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False, direction='in')
    ax14.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False, direction='in')
    ax15.tick_params(axis='y', labelleft=False, labelright=True, right=True, left=False, direction='in')
    
def sky_plot(x, y, frame='galactic', projection='aitoff', density=False, ax=None, xlimits=None, ylimits=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5), subplot_kw={'projection': projection})
        
    if projection == 'aitoff':        
        coords = SkyCoord(x, y, unit='degree', frame=frame)
        if frame == 'galactic':
            x = -coords.l.wrap_at(180 * u.deg).radian
            y = coords.b.radian
        elif frame == 'icrs':
            x = coords.ra.wrap_at(180 * u.deg).radian
            y = coords.dec.radian
        
        if density is True:
            density_scatter(x, y, bins = 500, ax=ax, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)
        
        if frame == 'galactic':
            # Convert the longitude values in right ascension hours
            ax.set_xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, \
                                          30, 60, 90, 120, 150]),
                        labels=['10h', '8h', '6h', '4h', '2h', '0h', \
                                '22h', '20h', '18h', '16h', '14h'])
            
            # Plot the labels and the title
            ax.set_title("Galactic coordinates" , x = 0.5, y = 1.1, fontsize=15, weight='bold')
            ax.set_xlabel('l', fontsize=15)
            ax.set_ylabel('b', fontsize=15)  
        elif frame == 'icrs':
            # Plot the labels and the title
            ax.set_title("ICRS" , x = 0.5, y = 1.1, fontsize=15, weight='bold')
            ax.set_xlabel('ra', fontsize=15)
            ax.set_ylabel('dec', fontsize=15)
        
        
        # Grid and legend, limits
        if xlimits is not None:
            ax.set_xlim(xlimits[0], xlimits[1])
        if ylimits is not None:
            ax.set_ylim(ylimits[0], ylimits[1])
        ax.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1,1.11))
        plt.grid(True, zorder=5)
    
    else:
        if density is True:
            plot = density_scatter(x, y, bins = 500, ax=ax, **kwargs)
            cb = plt.colorbar(plot, pad = 0.037, ax=ax)
            cb.set_label("Sources per bin", fontsize = 14, labelpad = -53)
        else:
            ax.scatter(x, y, **kwargs)
            
        if frame == 'galactic':
            #Labels and title
            ax.set_title("Galactic coordinates" , fontsize=15, weight='bold')
            ax.set_xlabel('l [deg]', fontsize = 15)
            ax.set_ylabel('b [deg]', fontsize = 15)  
        elif frame == 'icrs':
            #Labels and title
            ax.set_title("ICRS", fontsize=15, weight='bold')
            ax.set_xlabel("ra [deg]", fontsize = 15)
            ax.set_ylabel("dec [deg]", fontsize = 15)

        ax.tick_params(axis='both', which='major', labelsize=15)
        
        if xlimits is not None:
            ax.set_xlim(xlimits[0], xlimits[1])
        if ylimits is not None:
            ax.set_ylim(ylimits[0], ylimits[1])
        ax.legend(loc='upper center', fontsize=11, shadow=True)
        ax.grid(False)
         
def spectral_clas():
    low_res = [(3833,4730), (4761,5267), (5796,5996), (6530, 6850), (8143,8766)] #(4000,4730) 6463,6663
    high_res = [(3920,3980), (4080,4160), (4310,4400), (4465,4490), (4840,4930), (5870,5910), 
                (6540,6685), (8475,8675)]
    
    cafos_dict = {'2030965725082217088':'0200', '2074693061975359232':'0201',
                  '2164630463117114496':'0202', '2206767944881247744':'0203',
                  '1870955515858422656':'0204', '2013187240507011456':'0205',
                  '2060841448854265216':'0206', '2166378312964576256':'0207',
                  '2200433413577635840':'0208'}
    fies = ['4076568861833452160', '4299904519833646080', '2200433413577635840']
    lamost = ['187219239343050880'] #x3 low-res different epochs
    hermes = ['6123873398383875456'] #x2 different epochs
    
    savepath='cool_plots/Thesis_figures/spectra_clas'
    
    for name in cafos_dict:
        unit = cafos_dict.get(name)
        fig = plt.figure(constrained_layout=True, figsize=(12, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.1, hspace=0.25)
        ax0 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,0])
        ax5 = fig.add_subplot(gs[2,1])
        axes=[ax2, ax3, ax4, ax5]
        
        su.cafos_spectra(f'data/cafos_spectra/spectra1D_dswfz_uniB_{unit}.txt', 'data/TFM_table.csv', 
                          xrange=low_res[0], lines_file='data/spectral_lines.txt', priority=['1', 'raman'], 
                          plot=False, dered='fitz', ax=ax0)
        ax0.tick_params(axis='both', direction='in')
        ax0.set_ylabel('')
        ax0.set_xlabel('')
        ax0.set_title('Gaia '+name+' (CAFOS)', fontsize=16, weight='bold')
        
        ylim = ax0.get_ylim()
        minimum_flux=max(ylim[0], 0)
        ax0.set_ylim(bottom=minimum_flux, top=ylim[1]+ylim[1]*0.2)
        
        for ax, xrange in zip(axes, low_res[1:]):
            su.cafos_spectra(f'data/cafos_spectra/spectra1D_dswfz_uniB_{unit}.txt', 'data/TFM_table.csv', 
                              xrange=xrange, lines_file='data/spectral_lines.txt', priority=['1', 'raman'], 
                              plot=False, dered='fitz', ax=ax)
            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('')
            
        # fig.suptitle('Gaia '+name, fontsize=16, weight='bold')
        fig.text(0.5, 0.07, r'Rest wavelength [$\AA$]', ha='center', va='center', fontsize=15)
        fig.text(0.07, 0.5, r'Flux [$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$]', ha='center', 
                  va='center', rotation='vertical', fontsize=15)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{savepath}/CAFOS_{name}.png', bbox_inches = "tight", format = "png")
        plt.close()
    
    lamost_files=['spec-55977-GAC_080N37_V1_sp03-241.fits', 'spec-56687-GAC080N37V1_sp03-241.fits', 'spec-56948-HD052351N362354B_sp14-213.fits']
    color=['k', 'blue', 'orange']
    dates=['19-02-2012', '29-01-2014', '17-10-2014']
    for file, col, date in zip(lamost_files, color, dates):
        name = lamost[0]
        fig = plt.figure(constrained_layout=True, figsize=(12, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.15, hspace=0.25)
        ax0 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,0])
        ax5 = fig.add_subplot(gs[2,1])
        axes=[ax2, ax3, ax4, ax5]
        
        directory = 'data/lamost_spectra'
        su.lamost_spectra(f'{directory}/{file}', 'data/TFM_table.csv', 
                          xrange=low_res[0], lines_file='data/spectral_lines.txt', priority=['1', 'raman'],
                          dered=True, plot=False, ax=ax0, color=col)
        
        
        ax0.tick_params(axis='both', direction='in')
        ax0.set_ylabel('')
        ax0.set_xlabel('')
        ax0.set_title('Gaia '+name+' (LAMOST low-res)'+f'\n{date}', fontsize=16, weight='bold')
        # ax0.text(0.5, 0.98, date, fontsize=15, ha='center', va='center')
        
        ylim = ax0.get_ylim()
        ax0.set_ylim(bottom=ylim[0]-100, top=ylim[1]+1200)
        
        for ax, xrange in zip(axes, low_res[1:]):
            su.lamost_spectra(f'{directory}/{file}', 'data/TFM_table.csv', 
                              xrange=xrange, lines_file='data/spectral_lines.txt', priority=['1', 'raman'],
                              dered=True, plot=False, ax=ax, color=col)
            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('')
            ylim = ax.get_ylim()
            ax.set_ylim(bottom=ylim[0]-100, top=ylim[1]+500)
            
        # fig.suptitle('Gaia '+name, fontsize=16, weight='bold')
        fig.text(0.5, 0.07, r'Rest wavelength [$\AA$]', ha='center', va='center', fontsize=15)
        fig.text(0.06, 0.5, r'Flux [$ergs \: cm^{-2} \: s^{-1} \: \AA^{-1}$]', ha='center', 
                  va='center', rotation='vertical', fontsize=15)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{savepath}/LAMOST_{name}_{date}.png', bbox_inches = "tight", format = "png")
        plt.close()
        
    directory = 'data/HERMES_spectra'
    name='6123873398383875456'
    hermes_files=[f'{directory}/{name}_11-5.txt', f'{directory}/{name}_13-5.txt']
    color=['k', 'blue']
    dates=['11-05-2024', '13-05-2024']
    for file, col, date in zip(hermes_files, color, dates):
        
        fig = plt.figure(constrained_layout=True, figsize=(12, 15))
        gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.15, hspace=0.25)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,0])
        ax5 = fig.add_subplot(gs[2,1])
        ax6 = fig.add_subplot(gs[3,0])
        ax7 = fig.add_subplot(gs[3,1])
        axes=[ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        
        # ylim = ax0.get_ylim()
        # ax0.set_ylim(bottom=ylim[0]-100, top=ylim[1]+1200)
        
        if col == 'blue':
            alpha=0.5
        else:
            alpha=1
            
        for ax, xrange in zip(axes, high_res):
            spec = pd.read_csv(file, sep=' ')
            su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=xrange, Av=0.21,
                        lines_file='data/spectral_lines.txt', priority=['1', 'raman', '2', '3'],
                        plot=False, ax=ax, color=col, alpha=alpha)
            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('')
            # ylim = ax.get_ylim()
            # ax.set_ylim(bottom=ylim[0]-100, top=ylim[1]+500)
        
        fig.suptitle('Gaia '+name+' (HERMES)'+f'\n{date}', fontsize=16, weight='bold', y=0.93)
        # fig.suptitle('Gaia '+name, fontsize=16, weight='bold')
        fig.text(0.5, 0.09, r'Rest wavelength [$\AA$]', ha='center', va='center', fontsize=15)
        fig.text(0.07, 0.5, r'Flux [number of counts]', ha='center', 
                  va='center', rotation='vertical', fontsize=15)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{savepath}/HERMES_{name}_{date}.png', bbox_inches = "tight", format = "png")
        plt.close()
    
    av = [1.11, 0.71, 3.7]
    for name, AV in zip(fies, av):
        
        fig = plt.figure(constrained_layout=True, figsize=(12, 15))
        gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.15, hspace=0.25)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,0])
        ax5 = fig.add_subplot(gs[2,1])
        ax6 = fig.add_subplot(gs[3,0])
        ax7 = fig.add_subplot(gs[3,1])
        axes=[ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        
        # ylim = ax0.get_ylim()
        # ax0.set_ylim(bottom=ylim[0]-100, top=ylim[1]+1200)
        
        for ax, xrange in zip(axes, high_res):
            spec = pd.read_csv(f'data/FIES-M_spectra/{name}.txt', sep=' ')
            su.spectrum(spec['wavelength'], spec['flux'], title='', xrange=xrange, Av=AV,
                        lines_file='data/spectral_lines.txt', priority=['1', 'raman', '2', '3'],
                        plot=False, ax=ax, color='k')
            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('')
            ylim = ax.get_ylim()
            ax.set_ylim(top=ylim[1]+ylim[1]*0.3)
        
        fig.suptitle('Gaia '+name+' (FIES med-res)', fontsize=16, weight='bold', y=0.91)
        # fig.suptitle('Gaia '+name, fontsize=16, weight='bold')
        fig.text(0.5, 0.09, r'Rest wavelength [$\AA$]', ha='center', va='center', fontsize=15)
        fig.text(0.07, 0.5, r'Flux [W nm$^{-1}$ m$^{-2}$]', ha='center', 
                 va='center', rotation='vertical', fontsize=15)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{savepath}/FIES_{name}.png', bbox_inches = "tight", format = "png")
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
    ax.set_xlabel('Phase', fontsize=15)
    if yunits=='mag':
        ax.invert_yaxis()
        ax.set_ylabel(r'Magnitude', fontsize=15)
    elif yunits=='flux':
        ax.set_ylabel(r'Flux', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # plt.suptitle(name, fontsize=20, weight='bold')
    if ((24*60)/best_freq)<60:
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {(24*60) / best_freq:.3f}' + r'$ \, min$',
                     fontsize=15, weight='bold')
    elif ((((24*60)/best_freq)>60) and (((24*60)/best_freq)<(24*60))):
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {24 / best_freq:.3f}' + r'$ \, hr$',
                     fontsize=15, weight='bold')
    else:
        ax.set_title(f'{name}\nFreq: {best_freq:.4f}'+ '$d^{-1}$'+f'; period: {1/best_freq:.3f}' + r'$ \, d$',
                     fontsize=15, weight='bold')
   


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
        fig = plt.figure(figsize=(5,2.9*len(t_list)), facecolor=facecolor)
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
        ax1.text(0.02,0.93, filt_list[0], fontsize=14, transform = trans, style='italic')
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
                ax1.text(0.02,0.93, filt_list[i], fontsize=14, transform = trans, style='italic')
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
                ax.text(0.02,0.93, filt_list[i], fontsize=14, transform = trans, style='italic')
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
                ax.text(0.02,0.93, filt_list[i], fontsize=14, transform = trans, style='italic')
        
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
    plt.savefig(f'cool_plots/Thesis_figures/folds/{name}.png', bbox_inches="tight", format="png")
    plt.close()
    
def fold_lc_grid():
    
    dictionary = {'2166378312964576256':72.427805, '5524022735225482624':7.988562, '2060841448854265216':60.545008,
                  '2175699216614191360':4.769977, '2206767944881247744':2*4.625127/24, '5880159842877908352':63.541328, 
                  '460686648965862528':3.7606215/24, '5965503866703572480':7.321043/24, '4076568861833452160':56.90033, 
                  '5323384162646755712':2.792011, '2006088484204609408':2.5960645, '2061252975440642816':2.139183, 
                  '6123873398383875456':19.0433664/24, '5311969857556479616':13.7444267, '6228685649971375616':1.273980,
                  '1870955515858422656':46.211840}
    
    ztf = os.listdir('IRSA_ZTF_lightcurves_std')
    for i, fil in enumerate(ztf):
        ztf[i] = fil.split('.')[0]
    asas = os.listdir('ASAS-SN_lightcurves_std')
    for i, fil in enumerate(asas):
        asas[i] = fil.split('.')[0]
    atlas = os.listdir('ATLAS_lightcurves_std')
    for i, fil in enumerate(atlas):
        atlas[i] = fil.split('.')[0]
    
    for name in dictionary:
        
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
            
        period = dictionary.get(name)
        if name=='4076568861833452160':
            inss=['ASAS-SN']
        bulk_combine(name, inss, 1/(period), cycles=2, outliers='eb', facecolor=None)
        
def organie_lightcurves():
    
    lc_folders = ['ASAS-SN_lightcurves_std', 'ATLAS_lightcurves_std', 'IRSA_ZTF_lightcurves_std', 'NEOWISE_lightcurves_std']
    
    tiers_df = pd.read_csv('data/new_TFM_table.csv')    
    object_to_tier = dict(zip(tiers_df['Gaia source ID'], tiers_df['Tier']))
    
    # Ensure the output folder exists
    output_folder = 'LC_TIER'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in lc_folders:
        for filename in os.listdir(folder):
            object_name = filename.split('.')[0]
            T = object_to_tier[int(object_name)]
            
            tier_folder = os.path.join(output_folder, T)
            object_folder = os.path.join(tier_folder, object_name)
            
            os.makedirs(object_folder, exist_ok=True)
                    
            source_file = os.path.join(folder, filename)
            output_file_name = f'{folder}_object_name.csv'
            destination_file = os.path.join(object_folder, output_file_name)
            shutil.copy(source_file, destination_file)
                

   
def grid_lightcurves(tiers, binsize=0, fold=False):
    
    # Dictionary of colors for each instrument/filter combination
    color_dict = {
        ('ZTF', 'zg'):'g',
        ('ZTF', 'zr'):'r',
        ('ZTF', 'zi'):'goldenrod',
        ('IRSA_ZTF', 'zg'):'g',
        ('IRSA_ZTF', 'zr'):'r',
        ('IRSA_ZTF', 'zi'):'yellow',
        ('ASAS-SN', 'V'):'darkcyan',
        ('ASAS-SN', 'g'):'blue',
        ('ATLAS', 'o'):'orange',
        ('ATLAS', 'c'):'cyan',
        ('NEOWISE', 'W1'):'darkred',
        ('NEOWISE', 'W2'):'slategray'}
    
    dictionary = {'2166378312964576256':72.427805, '5524022735225482624':7.988562, '2060841448854265216':60.545008,
                  '2175699216614191360':4.769977, '2206767944881247744':2*4.625127/24, '5880159842877908352':63.541328, 
                  '460686648965862528':3.7606215/24, '5965503866703572480':7.321043/24, '4076568861833452160':56.90033, 
                  '5323384162646755712':2.792011, '2006088484204609408':2.5960645, '2061252975440642816':2.139183, 
                  '6123873398383875456':19.0433664/24, '5311969857556479616':13.7444267, '6228685649971375616':1.273980,
                  '1870955515858422656':46.211840}
    
    # Initialize lists to store object data across tiers
    all_objects = []
    all_labels = []

    # Gather all objects and their corresponding tier labels
    for tier in tiers:
        tier_folder = f'LC_TIER/{tier}'
        objects = os.listdir(tier_folder)
        all_objects.extend([(tier, obj) for obj in objects])
        all_labels.extend([f'({tier[0]})' for _ in objects])  # Use the first letter as label (D for Diamond, G for Gold)

    # Determine grid size
    num_objects = len(all_objects)
    cols = 3
    rows = (num_objects + cols - 1) // cols

    # Create the figure and axes for subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    # fig.suptitle(f"Light Curves for Tiers: {', '.join(tiers)}", fontsize=20)
    axes = axes.flatten()
    
    legend_elements = []
    added_labels = set()

    # Plot each object’s light curve
    for i, (tier, object_folder) in enumerate(all_objects):
        tier_folder = f'LC_TIER/{tier}'
        object_path = os.path.join(tier_folder, object_folder)
        csv_files = [f for f in os.listdir(object_path) if f.endswith(".csv")]
        
        dat = pd.read_csv(os.path.join(object_path, csv_files[0]))
        min_index = np.argmax(dat['mag'])
        t_start = dat['mjd'][min_index]

        # Initialize the subplot for the current object
        ax = axes[i]
        # ax.set_title(f'{object_folder} {all_labels[i]}')
        ax.text(0.01, 0.01, f'{object_folder}{all_labels[i]}-P:{dictionary[object_folder]:.2f}d', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='bottom', ha='left')
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Mag')
        
        # Plot each CSV file (light curve) for the object
        for csv_file in csv_files:
            data = pd.read_csv(os.path.join(object_path, csv_file))
            instrument = data['inst'][0]
            
            if instrument == 'NEOWISE':
                ms = 6
                cs = 4
                ew = 4
            else:
                ms = 3
                cs = 2
                ew = 2
                
            for filter_name in data['filter'].unique():
                filter_data = data[data['filter'] == filter_name]
                plot_color = color_dict.get((instrument, filter_name), 'black')
                #ax.scatter(filter_data['mjd'], filter_data['mag'], color=plot_color) #, label=f'{instrument} {filter_name}')
            # ax.scatter(data['mjd'], data['mag'])
            
                if instrument == 'NEOWISE':
                    t_observed = np.array(filter_data["mjd"])
                else:
                    t_observed = np.array(filter_data["bjd"])
                y_observed = np.array(filter_data['mag'])
                uncert = np.array(filter_data["magerr"])
                
                sigma = remove_outliers(filter_data['mag'], method='median')
                    
                if binsize > 0 and instrument!='NEOWISE':
                    t_observed, y_observed, t_err, uncert = bin_lightcurve(t_observed[sigma], y_observed[sigma], yerr=uncert[sigma], 
                                                                            binsize=binsize, mode="average")
                elif instrument == 'NEOWISE':
                    t_observed, y_observed, t_err, uncert = bin_lightcurve(t_observed[sigma], y_observed[sigma], yerr=uncert[sigma], 
                                                                            binsize=50, mode="average")
                else:
                    t_err = None
                
                if fold is True:
                    # min_index = np.argmax(y_observed)
                    t_observed = t_observed - t_start
                    if instrument=='NEOWISE':
                        continue
                    else:
                        best_freq = 1/dictionary[object_folder]
                        t_observed, y_observed, t_err, uncert = bin_lightcurve((t_observed[sigma] * best_freq) % 2, y_observed[sigma], yerr=uncert[sigma], 
                                                                                binsize=0.05, mode="average")
                        # ax.errorbar((t_observed * best_freq) % 2, y_observed, xerr=t_err, yerr=uncert, color=plot_color, #label=band, 
                                   # fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                        # ax.scatter((t_observed * best_freq) % 2, y_observed, c=plot_color, #label=band, 
                        #            marker='o', s=ms)
                        ax.scatter(t_observed, y_observed, c=plot_color, #label=band, 
                                   marker='o', s=ms)
                else:
                    if instrument=='NEOWISE':
                        ax.plot(t_observed, y_observed+2, linestyle='--', color=plot_color, alpha=0.5)
                        ax.errorbar(t_observed, y_observed+2, xerr=t_err, yerr=uncert, color=plot_color, #label=band, 
                                    fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                    else:
                        if binsize == 0:
                            t_observed = t_observed[sigma]
                            y_observed = y_observed[sigma]
                            uncert = uncert[sigma]
                        ax.errorbar(t_observed, y_observed, xerr=t_err, yerr=uncert, color=plot_color, #label=band, 
                                    fmt="o", capsize=cs, elinewidth=ew, markersize=ms, markeredgecolor='black', markeredgewidth=0.4)
                    
                # Add to legend if the combination hasn't been added yet
                if instrument=='NEOWISE':
                    label = f'{instrument} {filter_name}-band (+2 mag)'
                elif instrument=='IRSA_ZTF' or instrument=='ZTF':
                    instrument = 'ZTF'
                    if filter_name=='zr':
                        filter_name = 'r'
                    elif filter_name=='zg':
                        filter_name = 'g'
                    elif filter_name=='zi':
                        filter_name = 'i'
                    label = f'{instrument} {filter_name}-band'
                else:
                    label = f'{instrument} {filter_name}-band'
                if label not in added_labels:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_color, label=label, markersize=9))
                    added_labels.add(label)
    
        ax.tick_params(which='major', width=2, direction='out')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, direction='out')
        ax.tick_params(labelsize=15)
        ax.minorticks_on()
        ax.invert_yaxis()
        if fold is False:
            ax.set_xlim(56601, 60399)
            xmin, xmax = ax.get_xlim()    
            
            def xconv(x):
                tyear = Time(x, format="mjd")
                return tyear.jyear 
                
            xmin2 = xconv(xmin)
            xmax2 = xconv(xmax)
                
            ay2 = ax.twiny()
            ay2.minorticks_on()
            ay2.tick_params(which='major', width=2, direction='out')
            ay2.tick_params(which='major', length=7)
            ay2.tick_params(which='minor', length=4, direction='out')
            ay2.tick_params(labelsize=15)
            ay2.set_xlim([xmin2, xmax2])
            # ay2.set_xlabel('Year', fontsize=15)
            ay2.ticklabel_format(useOffset=False)
        
        if i < num_objects-3:
            ax.set_xticklabels([])  # Hide x-axis labels for all but the last row
        if i > 2 and fold is False:
            ay2.set_xticklabels([])
            
           
    fig.text(0.5, -0.02, 'MJD [days]', ha='center', fontsize=16)  # X-axis label
    if fold is False:
        fig.text(0.5, 0.96, 'Year', ha='center', fontsize=16)  # X-axis label
    fig.text(-0.02, 0.5, 'Apparent Magnitude', va='center', rotation='vertical', fontsize=16)  # Y-axis label
    if fold is True:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=15)
    else:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=15)

    # Hide any empty subplots (if there are more subplots than objects)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show or save the figure
    plt.show()  # or use plt.savefig(f"combined_tiers_lightcurves.png")
    plt.close()
    
        
    
#%% LIGHTCURVES
stacked_og_lc('2060841448854265216', ['IRSA_ZTF', 'ATLAS', 'ASAS-SN', 'NEOWISE'])
ax = plt.gca()
ax.set_title('Gaia 2060841448854265216', fontsize=16, weight='bold')
# plt.show()
plt.savefig('cool_plots/Thesis_figures/lightcurves.png', bbox_inches = "tight", format = "png")

#%% FINKER OUTPUT
tubla = pd.read_csv('IRSA_ZTF_lightcurves_std/2060841448854265216.csv')
t = np.array(tubla['bjd'][tubla['filter']=='zg'])
y = np.array(tubla['mag'][tubla['filter']=='zg'])
yerr = np.array(tubla['magerr'][tubla['filter']=='zg'])

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
freq = np.linspace(0.001,5,30000)
_, _ = fink.Finker_mag(t, y, yerr, freq, show_plot=False, calc_error=False, ax1=ax1, ax2=ax2)
axes_params(ax1, title='', xlabel=r'Frequency [$d^{-1}$]', ylabel='Squared Residuals', xlim=None, ylim=None, legend=False,
                title_size=14, label_size=15)
axes_params(ax2, title='', xlabel=r'Phase', ylabel='Magnitude', xlim=None, ylim=None, legend=False,
                title_size=14, label_size=15)
fig.suptitle('Gaia 2060841448854265216', fontsize=16, weight='bold')
# plt.show()
plt.savefig('cool_plots/Thesis_figures/FINKER_output.png', bbox_inches = "tight", format = "png")

#%% SKYPLOTS
random = Table.read('data/10mili.vot', format="votable") #10mili.vot, random_sources-result.vot
random = random.to_pandas()

mysources = Table.read('data/70_targets_coord.vot', format="votable")
mysources = mysources.to_pandas()

sky_plot(random['ra'], random['dec'], projection=None, frame='icrs', density=True, 
            s=2, cmap='plasma', norm='log', alpha=0.5, zorder=1)
ax1 = plt.gca()
sky_plot(mysources['ra'][mysources['phot_g_mean_mag']<13], mysources['dec'][mysources['phot_g_mean_mag']<13],
            projection=None, frame='icrs', c='w',marker='*', edgecolors='k', s=180, label="Mag<13", ax=ax1, zorder=6)
sky_plot(mysources['ra'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
            mysources['dec'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)],
            projection=None, frame='icrs', c='w',marker='*', edgecolors='k',s=100, label="13<Mag<15", ax=ax1, zorder=6)
sky_plot(mysources['ra'][mysources['phot_g_mean_mag']>15], mysources['dec'][mysources['phot_g_mean_mag']>15],
            projection=None, frame='icrs', c='w',marker='*', edgecolors='k',s=40, label="Mag>15", ax=ax1, zorder=6)

# plt.tight_layout()
plt.show()
plt.close()

sky_plot(random['l'], random['b'], density=True, s=2, cmap='plasma', alpha=0.5, norm='log')
ax2 = plt.gca()
sky_plot(mysources['l'][mysources['phot_g_mean_mag']<13], mysources['b'][mysources['phot_g_mean_mag']<13], 
            c='w', edgecolors='k',marker='*',s=180, label="Mag<13", ax=ax2, zorder=6)
sky_plot(mysources['l'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
            mysources['b'][(mysources['phot_g_mean_mag']>13)&(mysources['phot_g_mean_mag']<15)], 
            c='w', edgecolors='k',marker='*',s=100, label="13<Mag<15", ax=ax2, zorder=6)
sky_plot(mysources['l'][mysources['phot_g_mean_mag']>15], mysources['b'][mysources['phot_g_mean_mag']>15], 
            c='w', edgecolors='k',marker='*',s=40, label="Mag>15", ax=ax2, zorder=6)

# plt.tight_layout()
plt.show()
plt.close()

#%% ALL CAFOS SPECTRA
all_low_spectra()
# plt.show()
# plt.savefig('cool_plots/Thesis_figures/all_cafos.png', bbox_inches = "tight", format = "png")
plt.tight_layout()
plt.show()
plt.close()

#%% LAMOST LOW RES

lamost_lowres()
# plt.show()
plt.savefig('cool_plots/Thesis_figures/lamost_lowres.png', bbox_inches = "tight", format = "png")
plt.close()

#%% HERMES

hermes()
# plt.show()
plt.savefig('cool_plots/Thesis_figures/hermes_na_doublet.png', bbox_inches = "tight", format = "png")
plt.close()

#%% HERMES VELOCITY

spectrum = pd.read_csv('data/HERMES_spectra/6123873398383875456_11-5.txt', sep=' ')
spectrum2 = pd.read_csv('data/HERMES_spectra/6123873398383875456_13-5.txt', sep=' ')

mask = (spectrum['wavelength']>5894)*(spectrum['wavelength']<5898)
mask2 = (spectrum['wavelength']>5888)*(spectrum['wavelength']<5892)
mask3 = (spectrum2['wavelength']>5894)*(spectrum2['wavelength']<5898)
mask4 = (spectrum2['wavelength']>5888)*(spectrum2['wavelength']<5892)

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(7, 7))

su.spec_velocity(5896, spectrum['wavelength'][mask], spectrum['flux'][mask], 
                 label=r'5896 $\AA$', color='r', ax=ax1,
                 site='lapalma', RA=212.0047986, DEC=-33.2619834, obs_time='2024-5-11')
su.spec_velocity(5890, spectrum['wavelength'][mask2], spectrum['flux'][mask2], 
                 label=r'5890 $\AA$', color='b', ax=ax1,
                 site='lapalma', RA=212.0047986, DEC=-33.2619834, obs_time='2024-5-11')
ax1.axvline(0, color='k', ls='--', alpha=0.8)
ax1.set_xlim(left=-60, right=60)
ax1.set_title('Observation date: 11-5-2024', fontsize=15)

su.spec_velocity(5896, spectrum2['wavelength'][mask], spectrum2['flux'][mask], 
                 label=r'5896 $\AA$', color='r', ax=ax2,
                 site='lapalma', RA=212.0047986, DEC=-33.2619834, obs_time='2024-5-11')
su.spec_velocity(5890, spectrum2['wavelength'][mask2], spectrum2['flux'][mask2], 
                 label=r'5890 $\AA$', color='b', ax=ax2,
                 site='lapalma', RA=212.0047986, DEC=-33.2619834, obs_time='2024-5-11')
ax2.axvline(0, color='k', ls='--', alpha=0.8)
ax2.set_xlim(left=-60, right=60)
ax2.set_title('Observation date: 13-5-2024', fontsize=15)
# ax.set_ylim(top=0.4)
plt.tight_layout()
# plt.show()
plt.savefig('cool_plots/Thesis_figures/hermes_na_doublet_vel.png', bbox_inches = "tight", format = "png")
plt.close()

#%% HALPHA VEL GRID

H_velocity_spec()
plt.tight_layout()
# plt.show()
plt.savefig('cool_plots/Thesis_figures/Halpha_velocity.png', bbox_inches = "tight", format = "png")
plt.close()

#%% SPECTRAL LINES CLASSIFICATION

spectral_clas()

#%%

fold_lc_grid()