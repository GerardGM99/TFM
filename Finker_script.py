# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:20:28 2024

@author: Gerard Garcia
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from FINKER.utils_FINKER import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from lightcurve_utils import remove_outliers


def Finker_mag(t_observed, y_observed, uncert, freq, show_plot=False, calc_error=False, ax1=None, ax2=None):
    '''
    Given a light curve (time, magnitude and magnitude error) uses FINKER to
    find the best match for the frequency of the variable. The error on the 
    found frequency can be calculated as well, but the computational time rises
    significantly. It also plots the folded light curve and the frequency 
    search (frequency vs squared residuals).
    
    
    FINKER github: https://github.com/FiorenSt/FINKER/tree/main
    FINKER paper: https://arxiv.org/abs/2312.05408
    
    Parameters
    ----------
    t_observed: array of doubles
        Array with the time values of the light curve.
    y_observed: array of doubles
        Array with the magnitude values of the light curve.
    uncert: array of doubles
        Array with the error on the magnitudes.
    freq: array of doubles
        Array with the frequencies that will be analyzed with FINKER.
    show_plot: boolean, optional
        If True, the plots are shown in the console.
    calc_error: boolean, optional
        If True, the error on the best frequency is calculated.

    Returns
    -------
    best_freq: double
        The best frequency found with FINKER.
    freq_err: double
        The best frequency error.
    return_fig: matplotlib figure
        The matplotlib figure with the frequency search plot and the folded
        light curve plot.

    '''
    # Creating a FINKER instance
    finker = FINKER()
    
    # Running a parallel FINKER search
    best_freq, freq_err, result_dict = finker.parallel_nonparametric_kernel_regression(
        t_observed=t_observed,
        y_observed=y_observed,
        freq_list=freq,
        uncertainties=uncert,
        show_plot=False,
        kernel_type='gaussian',
        regression_type='local_constant',
        bandwidth_method='custom',
        alpha=0.0618,
        verbose=0,
        n_jobs=-2,
        tight_check_points=5000, 
        search_width=0.01,
        estimate_uncertainties=calc_error, 
        bootstrap_width=0.01,
        n_bootstrap=500,
        bootstrap_points=500
    )
        
    phase, y_smoothed, x_eval_in_interval, y_estimates_in_interval, squared_residual,bw = finker.nonparametric_kernel_regression(t_observed=t_observed,
                                                                                                                                 y_observed=y_observed,
                                                                                                                                 uncertainties=uncert,
                                                                                                                                 freq=best_freq, kernel_type='gaussian',
                                                                                                                                 regression_type='local_constant',
                                                                                                                                 bandwidth_method='custom',
                                                                                                                                 alpha=0.0618,
                                                                                                                                 show_plot=show_plot,
                                                                                                                                 use_grid=False)
    
    ### PLOT WITH INNESTS
    ### NEW STYLE PLOT

    frequencies_kernel = list(result_dict.keys())
    objective_kernel = list(result_dict.values())
    objective_kernel = (objective_kernel-min(objective_kernel))/(max(objective_kernel)-min(objective_kernel))

    # Create a figure with two subplots
    if ax1 is None:
        return_fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # First subplot: Frequency plot with inset
    ax1.scatter(frequencies_kernel, objective_kernel, s=1, c='royalblue')
    ax1.axvline(x=best_freq, color='black', linestyle='--', lw=2)
    ax1.set_xlabel(r'Frequency [$d^{-1}$]', fontsize=18)
    ax1.set_ylabel('Squared Residuals', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    if calc_error:
        legend_handle_best_freq = Line2D([0], [0], marker='o', color='black', label=f'Best Freq.: {best_freq:.6f}' + r'$ \, d^{-1} \, \pm \,$' + f'{freq_err:.7f}' , markersize=7,lw=3)
    else:
        legend_handle_best_freq = Line2D([0], [0], marker='o', color='black', label=f'Best Freq.: {best_freq:.6f}' + r'$ \, d^{-1}$', markersize=7,lw=3)
    if ((24*60)/best_freq)<60:
        legend_handle_period = Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=f'Period: {(24*60)/best_freq:.6f}' + r'$ \, min$')
    elif ((((24*60)/best_freq)>60) and (((24*60)/best_freq)<(24*60))):
        legend_handle_period = Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=f'Period: {24/best_freq:.6f}' + r'$ \, hr$')
    else:
        legend_handle_period = Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=f'Period: {1/best_freq:.6f}' + r'$ \, d$')
    ax1.legend(handles=[legend_handle_best_freq,legend_handle_period], loc='lower right', fontsize=14, framealpha=1.0)

    # Inset for the first subplot
    axins = inset_axes(ax1, width="90%", height="90%", bbox_to_anchor=(0.45,0.31, 0.45, 0.45), bbox_transform=ax1.transAxes)
    axins.scatter(frequencies_kernel, objective_kernel, s=5, c='royalblue')
    axins.errorbar(best_freq, objective_kernel.min(), xerr=freq_err,fmt='o', color='black',
                    ecolor='black', elinewidth=3, capsize=4, markersize=5,markeredgewidth=3)
    axins.axvline(x=best_freq, color='black', linestyle='--', lw=2)
    axins.set_ylim(-0.005, 0.05)
    if calc_error:
        axins.set_xlim(best_freq-3*freq_err, best_freq+3*freq_err)
    else:
        axins.set_xlim(best_freq-0.00003, best_freq+0.00003)
    axins.set_yticks([])

    axins.tick_params(axis='x', labelsize=14)
    mark_inset(ax1, axins, loc1=4, loc2=2, fc="none", ec="black", lw=1,ls='--', alpha=0.2)
    mark_inset(ax1, axins, loc1=3, loc2=3, fc="none", ec="black", lw=1,ls='--', alpha=0.2)
    for spine in axins.spines.values():
         spine.set_linewidth(1.2)


    # Second subplot: Light curve
    ax2.errorbar((t_observed * best_freq) % 1, y_observed,yerr=uncert, fmt='o', ecolor='black', capsize=2, elinewidth=2,
                 markersize=2, markeredgewidth=2, alpha=0.8, color='black',zorder=0)
    ax2.plot(phase, y_smoothed, c='green',lw=3)
    ax2.set_xlabel('Phase', fontsize=18)
    ax2.set_ylabel(r'Magnitude', fontsize=18)
    ax2.invert_yaxis()
    ax2.tick_params(axis='both', which='major', labelsize=14)

    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    # sort_indices = np.argsort((t_observed * best_freq) % 1)
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(phase, y_observed[sort_indices], yerr=uncert[sort_indices], fmt='.',
    #              markersize=5, label='Observed with Uncertainty', zorder=0)
        
    # plt.plot(x_eval_in_interval, y_estimates_in_interval, 'r-', label='Smoothed', linewidth=2)
    # plt.xlabel('Phase')
    # plt.ylabel('Flux')
    # #plt.title("Phase-folded Light Curve")
    # plt.legend()
    # plt.gca().invert_yaxis()
    # #plt.savefig(savefile+'/phase_folded.png', bbox_inches = "tight", format = "png")
    # folded_figure = plt.gcf()
    
    
    # # Plot the results
    # periodo_figure, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(list(result_dict.keys()), list(result_dict.values()), label='Objective Value', s=1)
    # ax.axvline(x=best_freq, color='g', linestyle='--', label=f'Best Frequency: {best_freq:.7f}', lw=2)
    # #ax.axvline(x=1, color='black', linestyle='--', label=f'True Frequency: {1:.7f}', lw=2)
    # ax.set_xlabel('Frequency', fontsize=18)
    # ax.set_ylabel('Squared Residuals', fontsize=18)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc='upper left', fontsize=14)
    

    
    return best_freq, freq_err


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def use_finker(lc_files, freq_range=20, freq_step=50000, calc_error=False, outliers='median'):
    
    '''
    Given a directory with lightcurves, applies FINKER to find the best
    frequency for every lightcurve in every filter.

    Parameters
    ----------
    lc_files: string
        Path to the directory containing the lightcurves in csv format.
    freq_range: float, optional
        Value of the greatest frequency analyzed with FINKER.
    freq_step: float, optional
        Steps in the frequency range analyzed with FINKER.
    calc_error: boolean, optional
        If True, the error on the best frequency is calculated.
    outliers: string, optional
        The manner in which the outliers in magnitude/flux are removed. Either
        'iqr' (InterQuartile Range) or 'median' (None to not remove outliers).
        
    
    Returns
    -------
    final_df: pandas DataFrame (columns: name, filter, frequency, freq_err)
        DataFrame with the frequencies found for each source and each filter, 
        including the frequency error calculated with FINKER.
    '''
    
    data_files = os.listdir(lc_files) # List with the names of the lc files
    data_files_names_only = [file_name.split(".csv")[0] for file_name in data_files]
    freq = np.linspace(0.001,freq_range,freq_step) # Frequency range analyzed with FINKER
    # Dictionary to store the results of the best frequencies found with FINKER
    dictionary = {'name':[], 'instrument':[], 'filter':[], 'frequency':[], 'freq_err':[]} 
    
    # Loop for all the lc files in the given directory
    for i in range(len(data_files)):
        # Read specific lc file
        lc = pd.read_csv(f'{lc_files}/{data_files[i]}')
        available_filters=list(set(lc['filter']))
        available_filters=sorted(available_filters)
        # Create output paths 
        if i==0:
            ins = lc['inst'].iloc[0] # Set name of the instrument/catalog
            if not os.path.isdir(f'./{ins}_finker_plots'):
                os.makedirs(f'./{ins}_finker_plots')
            
            general_path = f'./{ins}_finker_plots/folded_lcs'
            if not os.path.isdir(general_path):
                os.makedirs(general_path)
            
            general_path_2 = f'./{ins}_finker_plots/folded_lcs_2'
            if not os.path.isdir(general_path_2):
                os.makedirs(general_path_2)
                
        particular_path = f'./{ins}_finker_plots/{data_files[i]}'
        if not os.path.isdir(particular_path):
            os.makedirs(particular_path)
            
        if i%5==0:
            print(f'Starting with file {data_files[i]}')  
        # Loop for each filter 
        for band in available_filters:            
            # Remove outliers
            sigma = remove_outliers(lc['mag'].loc[lc['filter']==band], method=outliers, n_std=5)
            
            # Variables to input into the FINKER script
            t_observed = np.array(lc['bjd'].loc[lc['filter'] == band])[sigma]
            y_observed = np.array(lc['mag'].loc[lc['filter'] == band])[sigma]
            uncert = np.array(lc['magerr'].loc[lc['filter'] == band])[sigma]
            
            
            # FINKER function which calculated the value of the best frequency,
            # its error, and returns the phase folded plot and the frequency
            # search plot as well
            best_freq, freq_error, return_figure = Finker_mag(t_observed, 
                                                              y_observed, 
                                                              uncert, 
                                                              freq,
                                                              show_plot=False,
                                                              calc_error=calc_error)
            
            # Append the found values to the dictionary
            dictionary['name'].append(data_files_names_only[i])
            dictionary['instrument'].append(ins)
            dictionary['filter'].append(band)
            dictionary['frequency'].append(best_freq)
            dictionary['freq_err'].append(freq_error)
                        
            # Save the figures returned by the FINKER script 
            # (phase folded light curve and the frequency search plot)
            plt.ioff()
            return_figure.suptitle(f'Frequency search and Phase-folded Light Curve (filter: {band})', fontsize='xx-large')
            return_figure.savefig(f'{particular_path}/folded_{band}.png')
            if band == available_filters[0]:
                return_figure.savefig(f'{general_path}/{data_files_names_only[i]}.png')
            if len(available_filters)>1:
                if band == available_filters[1]:
                    return_figure.savefig(f'{general_path_2}/{data_files_names_only[i]}.png')
            plt.close()

            # plt.figure(periodo_figure.number)
            # plt.title(f'Frequency search (filter: {band})')
            # periodo_figure.savefig(f'{particular_path}/freq_search_{band}.png')
            # plt.close()
            
        # Plotting the light curve ############################
        # plt.figure(figsize=(10, 6))
        # plt.scatter(lc['mjd'].loc[lc['filtercode'] == 'zg'], 
        #             lc['mag'].loc[lc['filtercode'] == 'zg'],
        #             s=10, c='g')
        # plt.xlabel('Time')
        # plt.ylabel('Magnitude')
        # plt.title('Light Curve (filter: zg)')
        # plt.gca().invert_yaxis()  # Inverting y-axis for magnitude
        # plt.savefig(f'{particular_path}/light_curve_zg.png')
        # plt.close()
            
    
    # Return the dictionary with the best frequencies for each source and each filter        
    final_df = pd.DataFrame(dictionary)
    final_df.to_csv(f'data/{ins}_frequency_table.csv')
    return final_df
