# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 08:05:25 2021.

Meteorological Kriging

This code takes a meteorological variable timeseries coming from multiple
stations and interpolates them to a specified grid using the Ordinary Kriging
method.

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
from pykrige import OrdinaryKriging
# from progressbar import progressbar


# %% Kriging

# Kriging fitting
def kfit(kr_var, kr_coo, ts, ct):
    """
    Kriging fitting for a meteorological variable timeseries.

    kr_var = Dataframe with variable to apply Kriging,
    kr_coo = 'Lat/Y' and 'Lon/X' for the grid
    ts = Specific timestep column for the interpolation
    ct = Coordinates type: 'euclidean' or 'geodesic'

    Returns an object with the Kriging interpolation of the respective timestep
    """
    kr_da = kr_var.loc[ts].to_frame().T  # Dataframe containing data for the variable
    kr_df = pd.concat([kr_coo, kr_da], axis=0).dropna(axis=1)  # Remove NaN for Kriging calculations
    x = np.array(kr_df.loc['Lon/X'])  # Array with coordinates X or Longitude
    y = np.array(kr_df.loc['Lat/Y'])  # Array with coordinates Y or Latitude
    kr_dat = np.array(kr_df.loc[[ts]])  # Array with parameter values for the given coordinates

    krig_ord = OrdinaryKriging(x, y, kr_dat, variogram_model='spherical', nlags=15,
                               enable_plotting=False, verbose=False,
                               enable_statistics=False,
                               coordinates_type=ct, pseudo_inv=True)
    return krig_ord


def kint(var, st_coo, grid, name, ctyp):
    """
    Calculate interpolated values of a variable to a specific grid or points.

    var = DataFrame of original values.
    st_coo = Y and X station coordinates
    grid = Dataframe of coordinates for projection. First column Y, second column X
    name = String of variable name
    ctyp = Coordinates type: 'euclidean' or 'geodesic'
    Returns a Dataframe of the interpolated values to the grid

    """
    timesteps = list(var.index)
    x_p = grid.loc[grid.index[1]]
    y_p = grid.loc[grid.index[0]]
    new_grid = pd.DataFrame(index=var.index, columns=grid.columns)
    print('Starting Kriging interpolation')

    for step in timesteps:
        if timesteps.index(step) == len(timesteps)/2:
            print('Halfway there')
        if var.loc[step].mean() == 0 and np.nansum(var.loc[step]) == 0:
            new_grid.loc[step] = np.zeros(x_p.size)
        elif np.isnan(var.loc[step]).all():
            new_grid.loc[step] = np.full(x_p.size, np.nan)
        else:
            kr_result = kfit(var, st_coo, step, ctyp)
            var_int_co = kr_result.execute('points', x_p, y_p)[0]  # Variable interpolated column to add to the grid
            new_grid.loc[step] = var_int_co
    print('Finished Kriging interpolation')
    return new_grid
