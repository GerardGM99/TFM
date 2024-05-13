# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:53:43 2024

@author: xGeeRe
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec



fig = plt.figure(figsize=(10,14), layout='constrained')

gs = GridSpec(nrows=4, ncols=2, figure=fig)
ax1 = fig.add_subplot(gs[:-1, 0])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[3, :])

image1 = imread("SOURCES_FINAL/460686648965862528/460686648965862528_flc_3.761h_ztf.png")
ax1.imshow(image1, aspect='auto')
# ax1.set_title("Image 1")
ax1.axis('off')

image2 = imread("SOURCES_FINAL/460686648965862528/460686648965862528_bprp.png")
ax2.imshow(image2, aspect='auto')
ax2.axis('off')

image3 = imread("SPECTRA/GRVS/473575777103322496_RVS.png")
ax3.imshow(image3, aspect='auto')
ax3.axis('off')

image4 = imread("IRSA_ZTF_plots/187219239343050880.png")
ax4.imshow(image4, aspect='auto')
ax4.axis('off')

image5 = imread("SPECTRA/CAFOS_spectra/2030965725082217088.png")
ax5.imshow(image5, aspect='auto')
ax5.axis('off')

# fig.suptitle("GridSpec")

plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

newton = pd.read_csv('data/x-ray/4XMM-DR13/NXSA-Results-1714938050811.csv')
data = newton[newton['EPIC_SOURCE_CAT.OBSERVATION_ID']==201200201]
ep2 = data['EPIC_SOURCE_CAT.EP_2_FLUX']
ep3 = data['EPIC_SOURCE_CAT.EP_3_FLUX']
ep4 = data['EPIC_SOURCE_CAT.EP_4_FLUX']
ep5 = data['EPIC_SOURCE_CAT.EP_5_FLUX']
# ep8 = data['EPIC_SOURCE_CAT.EP_8_FLUX']
fluxes = [ep2, ep3, ep4, ep5]
wavelenghts = [16.531, 8.2656, 3.8149, 1.5028]
newton_df = pd.DataFrame({'wavelength':wavelenghts, 'flux':fluxes})

vosa = pd.read_csv('data/SED/vosa_results_77360/objects/2083649030845658624/sed/2083649030845658624_sed.csv')

plt.figure(figsize=(8,8))
plt.scatter(vosa['Wavelength'], vosa['Wavelength']*vosa['Flux'], s=10, color='r')
plt.scatter(newton_df['wavelength'], newton_df['wavelength']*newton_df['flux'], s=10, color='r')
plt.xscale('log')
plt.yscale('log')
plt.show()
