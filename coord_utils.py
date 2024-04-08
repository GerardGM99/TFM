# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:12:31 2024

@author: xGeeRe
"""

#import os
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from astropy.table import Table, join_skycoord
#from astropy import table
#from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astropy.coordinates import match_coordinates_sky
import requests
from PIL import Image
from io import BytesIO
import pylab
import os
from scipy.interpolate import interpn

def sky_xmatch(table1, table2, radius, column_names, read=False):
    '''
    Sky cross match two tables. First table is the main one, second table only
    needs to contain coordinates (ra, dec) and a name/id for the sources.
    
    Parameters
    ----------
    table1: astropy.Table or string
        Main table. Needs to contain at least coordinates in (ra, dec). Either
        the astropy table directly or a path to the csv of the table.
    table2: astropy.Table or string
        Table to sky crossmatch table1 to. Needs to contain at least 
        coordinates in (ra, dec) and a name/id of the source. Either the 
        astropy table directly or a path to the csv of the table.
    radius: float
        Maximum separation allowed in arcseconds.
    column_names: list of strings
        List with the names of the columns containing the coordinates and the
        name/id of the sources in both table1 and table2, respectively. The
        first two items of the list should be the names for the coordinate
        columns in table1, the 3rd and 4th should be the names for the
        coordinate columns in table2, and 5th item should be the name of the
        column with the name/id of the sources. An example could be:
            column_names = ['ra', 'dec', 'ra', 'dec', 'DR3_source_id']
    read: boolean, optional
        False (default) if the astropy tables are given directly, True if the
        path to the tables are given and need to be read.

    Returns
    -------
    table1: astropy.Table
        Cross mathed table with the columns in table1 and an additional column
        with the cross matched names/ids in table2.

    '''
    # Read both files if read=True
    if read is True:
        table1 = Table.read(table1, format='ascii.csv')
        table2 = Table.read(table2, format='ascii.csv') #Table with the name of the sources
    
    # Names of the columns with RA, DEC and the NAME of the source
    ra_col1 = column_names[0]
    dec_col1 = column_names[1]
    ra_col2 = column_names[2]
    dec_col2 = column_names[3]
    source_col = column_names[4]
    
    # Convert the RA and Dec columns of both Tables to SkyCoord objects
    coords_table1 = SkyCoord(ra=table1[ra_col1], dec=table1[dec_col1], unit='deg')
    coords_table2 = SkyCoord(ra=table2[ra_col2], dec=table2[dec_col2], unit='deg')
    
    # Create additional column to put the names/ids with the correct format
    table1.add_column('', name='Name', index=0)
    convert_column = table1['Name'].astype('U25')
    table1.replace_column('Name', convert_column)
    
    # Maximum separation allowed
    max_sep = radius * u.arcsec  

    # Iterate over unique sources in table2
    for idx2, source_name in enumerate(table2[source_col]):
        source_coord = coords_table2[idx2]
        # Find matches in table1 for the current source in table2
        sep_constraint = coords_table1.separation(source_coord) < max_sep
        matching_rows = table1[sep_constraint]
        if len(matching_rows) > 0:
            table1['Name'][sep_constraint] = source_name
    
    # ANOTHER METHOD I SHOULD EXPLORE BUT I DON'T HAVE TIME. FOR NOW IT KINDA
    # WORKS BUT THE XMATCHED TABLE ONLY HAS COORDINATES AND NAME (IT'S NOT A
    # JOIN OF THE ORIGINAL TABLES)
    
    # table1 = Table.read('data/irsa_ztf.csv', format='ascii.csv')
    # table2 = Table.read('data/70_targets.csv', format='ascii.csv')

    # # Convert the RA and Dec columns of both Tables to SkyCoord objects
    # coords_table1 = SkyCoord(ra=table1['ra'], dec=table1['dec'], unit='deg')
    # coords_table2 = SkyCoord(ra=table2['ra'], dec=table2['dec'], unit='deg')

    # t1 = Table([coords_table1], names=['s'])
    # t2 = Table([coords_table2], names=['s'])
    # t2.add_column(table2['DR3_source_id'], name='Name', index=0)
    # t12 = table.join(t1, t2, join_funcs = {'s': join_skycoord(1 * u.arcsec)})
    # df = t12.to_pandas()
    
    
    return table1

#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# PS1 Image Cutout Service (https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service)
# Functions are from their Jupyter Notebook example. Added "draw_image".

def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    
    """Get grayscale image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def draw_image(ra, dec, name, size, directory='.', ext="pdf", plot=True, save=False):
    '''
    Plots the color cutout form Pan-STARRS.

    Parameters
    ----------
    ra : float
        Right ascension of the center of the cutout.
    dec : float
        Declination of the center of the cutout.
    name : string
        Name to give to the image.
    size : float
        Size of the image in pixels.
    directory : string, optional
        Directory where to save the image. The default is '.'.
    ext : string, optional
        Extension for the saved image. The default is "pdf".
    plot : bool, optional
        If True, show the plot. The default is True.
    save : bool, optional
        If True, save the plot. The default is False.

    Returns
    -------
    None.

    '''
    
    if dec>-30:
    # color image
        cim = getcolorim(ra,dec,size=size,filters="grizy")
        
        plt.figure(figsize=(8, 6))
        plt.grid(color='white', ls='solid')
        plt.imshow(cim,origin="upper")
        
        #Mark target in green
        ax = plt.gca()
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        plt.plot(0.5, 0.5, '+', c='green', markersize=10, markeredgewidth=2, markeredgecolor='green',
                 transform=trans)
        
        plt.title(name, fontsize=15, weight='bold')
        
        # Plot compass
        plt.plot([0.83,0.93], [0.05,0.05], 'w', lw=2, transform=trans)
        plt.plot([0.93,0.93], [0.05,0.15], 'w', lw=2, transform=trans)
        plt.text(0.78, 0.08, "E", transform = trans, fontdict={'color':'w'})#, fontdict={'fontsize':14})
        plt.text(0.88, 0.17, "N", transform = trans, fontdict={'color':'w'})#, fontdict={'fontsize':14})
        
        # Calculate RA and Dec ticks
        ra_ticks = [ra - 0.4*size*0.25/3600, ra - 0.15*size*0.25/3600, 
                    ra + 0.1*size*0.25/3600, ra + 0.35*size*0.25/3600]
        dec_ticks = [dec - 0.35*size*0.25/3600, dec - 0.1*size*0.25/3600, 
                     dec + 0.15*size*0.25/3600, dec + 0.4*size*0.25/3600]
        
        
        # Convert RA and Dec
        ticks = SkyCoord(ra=ra_ticks, dec=dec_ticks, frame='icrs', unit='deg')
        ra_ticks = ticks.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
        dec_ticks = ticks.dec.to_string(sep=':', precision=2, alwayssign=True, pad=True)
        
        
        ax.set_xticks([size*0.1, size*0.35, size*0.6, size*0.85])
        ax.set_yticks([size*0.85, size*0.6, size*0.35, size*0.1])
        ax.set_xticklabels([ticks[0].ra.to_string(unit=u.hourangle, sep=['$^h$','$^m$','$^s$'], precision=2, pad=True), 
                            '%.2f$^s$'%(ticks[1].ra.hms[2]), '%.2f$^s$'%(ticks[2].ra.hms[2]), '%.2f$^s$'%(ticks[3].ra.hms[2])])
        ax.set_yticklabels(['%.2f"'%(ticks[0].dec.dms[2]), '%.2f"'%(ticks[1].dec.dms[2]), '%.2f"'%(ticks[2].dec.dms[2]),
                            ticks[3].dec.to_string(sep=[r'$^\circ$', '\' ', '"'], precision=2, alwayssign=True, pad=True)])
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
    
        ax.set_xlabel('%.1f\''%(size*0.25/60))
        ax.set_ylabel('%.1f\''%(size*0.25/60))
    
        if plot:
            plt.show()
            plt.close()
            
        if save:
            pylab.savefig(os.path.join(directory, str(str(name)+'_color.%s'%ext)))
            pylab.close("all")
        
    else:
        print(f'No PANSTARS for {name}')
    
    
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


#Function to plot sources as a scatter plot with density
def density_scatter( x , y, sort = True, bins = 20, ax=None, **kwargs )   :
    '''
    Scatter plot colored by 2d histogram.

    Parameters
    ----------
    x : list of floats
        X axis values.
    y : list of floats
        Y axis values.
    sort : bool, optional
        If True, sort the points by density, so that the densest points are 
        plotted last. The default is True.
    bins : float, optional
        Number of bins for the np.histogram2d. The default is 20.
    ax : Axes object (matplotlib), optional
        Axes where to draw the density plot. The default is None.
    **kwargs :
        Additional parameters for matplotlib.pyplot.scatter.

    Returns
    -------
    out : Scatter plot (matplotlib)
        The density scatter plot.

    '''
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    out = ax.scatter( x, y, c=z, **kwargs )
    #plt.colorbar(label = "Sources per bin", location="left")
        
    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm))
    #cbar.ax.set_ylabel('Density')

    return out


#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def sky_plot(x, y, frame='galactic', projection='aitoff', density=False, ax=None, **kwargs):
    '''
    Coordinate plot

    Parameters
    ----------
    x : list of floats
        Coordinates of the x axis (ra or l).
    y : list of floats
        Coordinates of the y axis (dec or b)..
    frame : string, optional
        Frame of the (x,y) coordinates. The 'galactic' frame has (l,b) 
        coordinates, the 'icrs' frame has (ra, dec) coordinates. The default 
        is 'galactic'.
    projection : string, optional
        Subplot projection. The default is 'aitoff'.
    density : bool, optional
        If True, a density scatter plot is shown, using the 'density_scatter'
        function. The default is False.
    ax : Axes object (matplotlib), optional
        Axes where to draw the density plot. If None, a new axes is created. 
        The default is None.
    **kwargs :
        Additional parameters for matplotlib.pyplot.scatter.

    Returns
    -------
    None.

    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), subplot_kw={'projection': projection})
        
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
            ax.set_title("Skymap in galactic coordinates" , x = 0.5, y = 1.1, fontsize=19)
            ax.set_xlabel('l')
            ax.set_ylabel('b')  
        elif frame == 'icrs':
            # Plot the labels and the title
            ax.set_title("Skymap in ICRS" , x = 0.5, y = 1.1, fontsize=19)
            ax.set_xlabel('ra')
            ax.set_ylabel('dec')
        
        
        # Grid and legend
        ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1,1.11))
        ax.grid(True)
    
    else:
        if density is True:
            plot = density_scatter(x, y, bins = 500, ax=ax, **kwargs)
            cb = plt.colorbar(plot, pad = 0.037, ax=ax)
            cb.set_label("Sources per bin", fontsize = 19, labelpad = -75)
        else:
            ax.scatter(x, y, **kwargs)
            
        if frame == 'galactic':
            #Labels and title
            ax.set_title("Skymap in galactic coordinates" , fontsize=19)
            ax.set_xlabel('l (deg)', fontsize = 19)
            ax.set_ylabel('b (deg)', fontsize = 19)  
        elif frame == 'icrs':
            #Labels and title
            ax.set_title("Skymap in ICRS", fontsize=19)
            ax.set_xlabel("ra (deg)", fontsize = 19)
            ax.set_ylabel("dec (deg)", fontsize = 19)

        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.legend(loc='upper center', fontsize=12, shadow=True)
        ax.grid(True)