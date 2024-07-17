# -*- coding: utf-8 -*-
"""
@author: Gerard Garcia Moreno
"""

'''
cbs --> Bias
cth --> Dark
cdG / cdR --> Flat (g/r filter)
'''

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from ccdproc import ImageFileCollection #conda install -c astropy ccdproc
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
#from tea_utils import tea_avoid_astropy_warnings
from tea_utils import tea_imshow
from tea_utils import tea_ifc_statsummary
from tea_utils import SliceRegion2D
import numpy as np
import matplotlib.patches as patches
from astropy.nddata import CCDData
import ccdproc
from astropy.stats import mad_std
from tea_utils import tea_statsummary
from tqdm.notebook import tqdm
from astropy.nddata.nduncertainty import StdDevUncertainty
from ccdproc import ImageFileCollection
from tea_utils import tea_avoid_astropy_warnings
from tea_wavecal import TeaWaveCalibration
from tea_utils import SliceRegion1D
import astropy.units as u
from tea_wavecal import apply_wavecal_ccddata
from tea_wavecal import fit_sdistortion
from tea_wavecal import polfit_residuals_with_sigma_rejection
import os, glob
import numpy as np

DATADIR = 'calib'
def read_calibrations(DATADIR):
    directory=Path(DATADIR)
    selected_keywords = [ #fits header parameters to check
        'IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
            'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS', 'AIRMASS'
    ]
    
    ifc_all = ImageFileCollection(
        location=directory,
        glob_include='TJO*.fits',
        keywords=selected_keywords
    )
    
    summary_all = tea_ifc_statsummary(ifc_all, directory)
    return directory, ifc_all, summary_all

# files = os.listdir(directory)
# bias_files=[]
# dark_files=[]
# flat_files=[]
# for file in files:
#     if 'cbs' in file:
#         bias_files.append(file)
#     if 'cth' in file:
#         dark_files.append(file)
#     if 'cdG' in file or 'cdR' in file:
#         flat_files.append(file)

# Master bias
def create_master_bias(calibrations, science_images, show_master_bias=False):
    directory, ifc_all, summary_all = read_calibrations(calibrations)
    
    matches_bias= ['bias' in object_name.lower() for object_name in ifc_all.summary['OBJECT']]
    summary_bias = summary_all[matches_bias]
    
    list_bias = []
    for filename in summary_bias['file']:
        # filepath = f'{directory}/{filename}'
        filepath = directory / filename
        
        data = fits.getdata(filepath)
        header = fits.getheader(filepath)
        
        list_bias.append(CCDData(data=data, header=header, unit='adu'))
        
    num_bias = len(list_bias)
    print(f'Number of BIAS exposures: {num_bias}')
    
    master_bias = ccdproc.combine(
        img_list=list_bias,
        method='average',
        sigma_clip=True, sigma_clip_low_thres=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std
    )
    
    master_bias.write(f'{directory}/master_bias.fits', overwrite='yes')
    median_bias = np.median(master_bias.data)
    gain = 1 # Sensitivity at selected gain (e-/DN)
    readout_noise = np.mean(summary_bias['robust_std'])
    
    ## Check master bias
    if show_master_bias is True:
        _ = tea_statsummary(master_bias.data)
        fig, ax = plt.subplots(figsize=(15, 5))
        vmin, vmax = np.percentile(data, [5, 95])
        tea_imshow(fig, ax, master_bias, vmin=vmin, vmax=vmax)
        ax.set_title('Master BIAS')
        plt.show()
    
    # Read science image
    directory = Path(science_images)
    science_files = os.listdir(science_images)
    for filename in science_files:
        if filename.startswith('TJO'):
            filepath = directory / filename
            
            
            # read data and header from file
            data = fits.getdata(filepath)
            header = fits.getheader(filepath)
            print(filepath,header['OBJECT'])
            
            # create associated uncertainty
            uncertainty1 = (data - median_bias) / gain
            uncertainty1[uncertainty1 < 0] = 0.0   # remove unrealistic negative estimates
            uncertainty2 = readout_noise**2
            uncertainty = np.sqrt(uncertainty1 + uncertainty2)
            
            # create CCDData instance
            ccdimage = CCDData(
                data=data,
                header=header,
                uncertainty=StdDevUncertainty(uncertainty),
                unit='adu'
            )
            
            
            # subtract master_bias
            ccdimage_bias_subtracted = ccdproc.subtract_bias(ccdimage, master_bias)
            
            
            # rotate primary HDU and extensions
            # ccdimage_bias_subtracted.data = np.rot90(ccdimage_bias_subtracted.data, 3)
            # ccdimage_bias_subtracted.mask = np.rot90(ccdimage_bias_subtracted.mask, 3)
            # ccdimage_bias_subtracted.uncertainty.array = np.rot90(ccdimage_bias_subtracted.uncertainty.array, 3)    
            
            # update FILENAME keyword with output file name
            output_filename = f'bias_corrected_{filename}'
            ccdimage_bias_subtracted.header['FILENAME'] = output_filename
            
            # save result
            ccdimage_bias_subtracted.write(directory / output_filename, overwrite='yes')
    
    