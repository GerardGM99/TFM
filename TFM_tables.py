# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:17:00 2024

@author: xGeeRe
"""

from astropy.table import Table
import pandas as pd
import numpy as np
import os

def check_file_in_directories(id, directories):
    instruments_with_file = []
    for folder in directories:
        instrument = os.path.basename(folder).split('_')[0]  # Extracting the instrument part
        folder_files = os.listdir(folder)
        for file in folder_files:
            if file.startswith(f"{id}"):
                instruments_with_file.append(instrument)
                break  # No need to check further in this folder if a matching file is found
    return ', '.join(instruments_with_file)

table = pd.read_csv('data/70_targets_extended.csv')

out = pd.DataFrame()

out['Gaia source ID'] = table['source_id']
out['RA'] = round(table['ra'], 4)
out['DEC'] = round(table['dec'], 4)
out['Distance'] = round(table['dist50']*1000)
# out['$PM_{RA}$'] = round(table['pmra'], 2).astype(str)+' $\pm$ '+round(table['pmra_error'], 2).astype(str)
# out['$PM_{DEC}$'] = round(table['pmdec'], 2).astype(str)+' $\pm$ '+round(table['pmdec_error'], 2).astype(str)
out['G'] = round(table['phot_g_mean_mag'], 2)
out['$log_{10}(G_{err})$'] = round(table['log_g_err'], 2)
out['$M_G$'] = round(table['mg0'], 2)
out['BP-RP'] = round(table['bp_rp'], 2)
out['$A_G$'] = round(table['ag50'], 2)
out['$A_V$'] = round(table['av50'], 2)
out['RV'] = round(table['radial_velocity'], 1)
out['$RV_{error}$'] = round(table['radial_velocity_error'], 1)
out['W1-W4'] = round(table['w1mpro']-table['w4mpro'], 2)
out['H$\alpha_W$'] = round(table['halpha_w'], 3)
out['H$\beta_W$'] = round(table['hbeta_w'], 3)
# out['Gaia Var. Class'] = table['best_class_name']
# out['Simbad Class'] = table['main_type']+' ('+table['other_types']+')'

out = out.fillna('-')

df1 = out.iloc[:35].reset_index(drop=True)
df2 = out.iloc[35:].reset_index(drop=True)

astropy_tab = Table.from_pandas(out)
astropy_tab.write('data/TFM_table.tex', format='latex', overwrite=True)
astropy_tab.write('data/TFM_table.csv', format='ascii.csv', overwrite=True)

astropy_df1 = Table.from_pandas(df1)
astropy_df1.write('data/df1.tex', format='latex', overwrite=True)

astropy_df2 = Table.from_pandas(df2)
astropy_df2.write('data/df2.tex', format='latex', overwrite=True)

######################################################################################################################

new = pd.DataFrame()

new['Gaia source ID'] = table['source_id']

crowd = pd.read_csv('data/crowded.csv')
mapping_dict = dict(zip(crowd['name'], crowd['Crowded']))
new['Crowded'] = new['Gaia source ID'].map(mapping_dict)

lc_list = []
directories = ['ZTF_lightcurves_std', 'ASAS-SN_lightcurves_std', 'ATLAS_lightcurves_std', 'NEOWISE_lightcurves_std', 'TESS_lightcurves_std']
for ide in out['Gaia source ID']:
    inst = check_file_in_directories(ide, directories)
    lc_list.append(inst)    
new['Light curves'] = lc_list

spec_list = []
directories = ['SPECTRA/CAFOS_spectra', 'SPECTRA/LAMOST_spectra/LAMOST-M_spectra', 
               'SPECTRA/LAMOST_spectra/LAMOST-L_spectra', 'SPECTRA/GRVS', 'data/FIES-M_spectra',
               'data/HERMES_spectra']
for ide in out['Gaia source ID']:
    inst = check_file_in_directories(ide, directories)
    if inst == '':
        spec_list.append(None)
    else:
        spec_list.append(inst)    
new['Spectrum'] = spec_list


tier_mapping = {1: 'Diamond', 2: 'Gold', 3: 'Silver', 4: 'Bronze'}

my_class = pd.read_csv('data/my_class.txt', sep=' ')
mapping_xray = dict(zip(my_class['source_id'], my_class['X-ray']))
mapping_class = dict(zip(my_class['source_id'], my_class['My_class']))
mapping_period = dict(zip(my_class['source_id'], my_class['Period']))
mapping_tier = dict(zip(my_class['source_id'], my_class['Tier']))
new['X-ray cat'] = new['Gaia source ID'].map(mapping_xray)
# new['LC class'] = new['Gaia source ID'].map(mapping_class)
new['Period'] = new['Gaia source ID'].map(mapping_period)
# new['Gaia Var. Class'] = table['best_class_name']
# new['Simbad Class'] = table['main_type']+' ('+table['other_types']+')'
new['Tier'] = new['Gaia source ID'].map(mapping_tier).map(tier_mapping)


new = new.fillna('-')

df3 = new.iloc[:35].reset_index(drop=True)
df4 = new.iloc[35:].reset_index(drop=True)

astropy_new_tab = Table.from_pandas(new)
astropy_new_tab.write('data/new_TFM_table.tex', format='latex', overwrite=True)
astropy_new_tab.write('data/new_TFM_table.csv', format='ascii.csv', overwrite=True)

astropy_df3 = Table.from_pandas(df3)
astropy_df3.write('data/df3.tex', format='latex', overwrite=True)

astropy_df4 = Table.from_pandas(df4)
astropy_df4.write('data/df4.tex', format='latex', overwrite=True)

#####################################################################################################

clases = pd.DataFrame()

clases['Gaia source ID'] = table['source_id']

clases['LC class'] = new['Gaia source ID'].map(mapping_class)
clases['Gaia Var. Class'] = table['best_class_name']
clases['Simbad Class'] = table['main_type']+' ('+table['other_types']+')'

clases = clases.fillna('-')

df5 = clases.iloc[:35].reset_index(drop=True)
df6 = clases.iloc[35:].reset_index(drop=True)

astropy_clases = Table.from_pandas(clases)
astropy_clases.write('data/clases.tex', format='latex', overwrite=True)
astropy_clases.write('data/clases.csv', format='ascii.csv', overwrite=True)

astropy_df5 = Table.from_pandas(df5)
astropy_df5.write('data/df5.tex', format='latex', overwrite=True)

astropy_df6 = Table.from_pandas(df6)
astropy_df6.write('data/df6.tex', format='latex', overwrite=True)