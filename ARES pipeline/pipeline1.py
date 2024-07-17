# -*- coding: utf-8 -*-
"""
@author: Gerard Garcia Moreno
"""

from PIL import Image
import numpy as np
from astropy.io import fits

import pylab as pl
pl.rcParams['image.origin'] = 'lower'

# Read the fits file and convert the data into an image
spectrum_filename = input("Enter file name: ") #"spectra/TJO2460424.60187_S_imr.fits"
hdul = fits.open(spectrum_filename)
data = hdul[0].data
data = data - np.min(data)
data = data / np.max(data) * 255
data = data.astype(np.uint8)
image_data = Image.fromarray(data)
hdul.close()

# Show the spectrum
pl.figure(figsize=(10,4))
pl.imshow(image_data, cmap='gray', vmax=2)
pl.show()

# Select y range
bottom_yval = float(input("ymin: "))
top_yval = float(input("ymax: "))

image_data = image_data.crop((0, bottom_yval, image_data.width, top_yval))

# Show the spectrum (again)
pl.figure(figsize=(10,6))
pl.imshow(image_data, cmap='gray', vmax=2, aspect='auto')
pl.show()

# Select y range
bottom_yval = float(input("ymin: "))
top_yval = float(input("ymax: "))

image_data = image_data.crop((0, bottom_yval, image_data.width, top_yval))

# Find bad pixels

yvals = np.argmax(image_data, axis=0)
xvals = np.arange(image_data.width)

pl.figure(figsize=(10,7))
pl.plot(xvals, yvals, 'x')
pl.ylabel("Argmax trace data")
pl.xlabel("X position")
pl.show()

# Prompt the user to enter the first number
bottom_yval = float(input("Enter lower yval: "))

# Prompt the user to enter the second number
top_yval = float(input("Enter upper yval: "))

# Print the numbers to verify input
# print(f"The first number is: {bottom_yval}")
# print(f"The second number is: {top_yval}")

# Show bad pixels
bad_pixels = (yvals < bottom_yval) | (yvals > top_yval)

pl.figure(figsize=(10,7))
pl.plot(xvals, yvals, 'x')
pl.plot(xvals[bad_pixels], yvals[bad_pixels], 'rx')
pl.ylabel("Argmax trace data")
pl.xlabel("X position")
pl.show()