# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:21:03 2024

@author: xGeeRe
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

table = pd.read_csv('data/TESS_TIC_file.csv')

for name, tic in zip(table['DR3_source_id'], table['TIC']):
    search_result = lk.search_lightcurve('TIC %s' %tic)
    lc = search_result.download_all()
    if lc is not None:
        for i in range(len(lc)):
            try:
                lc_clean = lc[i].remove_nans().remove_outliers()
            except:
                lc_clean = lc[i]
            author = lc_clean.author
            sector = search_result[i].mission[0].split(' ')[2]
            lc_clean.to_csv(f'TESS_lightcurves/{name}_S{sector}_{author}.csv', overwrite=True)


        
# def TESS_QLP(lc):
#     for i in range(len(lc)):
#         lc[i].remove_outliers()
#         try:
#             lc[i].remove_outliers().scatter(column='kspsap_flux')
#         except:
#             lc[i].remove_outliers().scatter(column='det_flux')

# lc.scatter(column='kspsap_flux')
# lc.scatter(column='det_flux')

# lc.to_periodogram('bls').plot()

# period = lc.to_periodogram('bls').period_at_max_power
# # lc.fold(period).scatter(column='kspsap_flux')
# lc.fold(period).scatter(column='det_flux')

# plt.figure()
# plt.scatter(lc['time'].value, lc['sap_flux'])
# plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt

file1 = pd.read_csv('TESS_lightcurves_std/5311969857556479616_S62_QLP.csv')
# file2 = pd.read_csv('TESS_lightcurves/1870955515858422656_S15_QLP.csv')


plt.figure()
# plt.scatter(file2['time'][mask]/(19.0433664/24)%1, file2['kspsap_flux'][mask], s=5)
plt.scatter(file1['bjd'], file1['flux'], s=5)
plt.show()

#%%

import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve.correctors import DesignMatrix
from lightkurve.correctors import RegressionCorrector

#All in one go (plus removing scattered light, allowing for the large offset from scattered light)

tess = pd.read_csv("data/TESS_TIC_file.csv")

target = "TIC "+str(tess["TIC"][6])

tpf = lk.search_tesscut(target)[0].download(cutout_size=(50, 50))

# Make an aperture mask and an uncorrected light curve
aper = tpf.create_threshold_mask()
uncorrected_lc = tpf.to_lightcurve(aperture_mask=aper)

# Make a design matrix and pass it to a linear regression corrector
dm = DesignMatrix(tpf.flux[:, ~aper], name='regressors').pca(5).append_constant()
rc = RegressionCorrector(uncorrected_lc)
corrected_ffi_lc = rc.correct(dm)

# Optional: Remove the scattered light, allowing for the large offset from scattered light
corrected_ffi_lc = uncorrected_lc - rc.model_lc + np.percentile(rc.model_lc.flux, 5)


ax = uncorrected_lc.plot(label='Original light curve')
corrected_ffi_lc.plot(ax=ax, label='Corrected light curve')
plt.show()

corrected_ffi_lc.plot(label='Corrected light curve')
plt.show()