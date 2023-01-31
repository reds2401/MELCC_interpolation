# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 08:05:25 2021.

Meteorological Kriging
This code takes a set of meteorological timeseries coming from multiple
stations and interpolates them to a specified grid using the Ordinary Kriging
method. The meteorological variables come from the stations of the Ministère
de l'environement of Quebec and the objective grid is to be used for
modelisation in hydrotel.

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj as proj
from pykrige import OrdinaryKriging
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

import Kriging

# %% Hydrotel files

# Import hydrotel meteorological grid file with coordinates
folder1 = "C:/Users/reds2401/Documents/HYDROTEL/Hydrotel_030101/meteo_original/"
# folder1 = "D:/Sergio/OneDrive - USherbrooke/Research_project/HYDROTEL/Hydrotel_030101_000/meteo/"
file1 = "station.stm"
headers1 = ["ID", "Lon", "Lat", "Alt", "Type", "Text1", "Text2"]
hyd_coord = pd.read_csv(folder1 + file1, encoding='latin-1', sep=" ",
                        names=headers1, skiprows=3).set_index('ID').drop(['Type', 'Text1', 'Text2'], axis=1)

# Reading Hydrotel grid files
headers2 = ['Date', 'Hour', 'TMax', 'Tmin', 'PT']  # Defining column names
hyd_gr_dates = pd.read_csv(folder1 + hyd_coord.index[0] + '.met', sep='\s+', names=headers2, skiprows=[0],
                           usecols=['Date', 'Hour'], dtype=object)
hyd_gr = pd.DataFrame()  # Initializing Hydrotel grid dataframe

for hgsta in hyd_coord.index:
    hgs_df = pd.read_csv(folder1 + hgsta + '.met', sep="\s+", names=headers2, skiprows=[0])  # Hydrotel station grid df
    hyd_gr[hgsta + '-TMax'] = hgs_df['TMax']
    hyd_gr[hgsta + '-Tmin'] = hgs_df['Tmin']
    hyd_gr[hgsta + '-PT'] = hgs_df['PT']

# Join dates with data to remove fragmentation
hyd_gr_df = pd.concat([hyd_gr_dates, hyd_gr], axis=1).set_index(['Date', 'Hour'])

# %% Info-Climat files: stations ID

# Import file with stations coordinates
folder2 = "C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Info/20210603-Meteo(2019-2021)/"
file2 = "Stations_Nicolet.csv"
# Dataframe with station coordinates
st_info = pd.read_csv(folder2 + file2, sep='\s+').set_index('NO_STATION').drop(['NOM_STATION'], axis=1)
st_info.columns = ['Lat', 'Lon', 'Alt']

# %% Info-Climat files: Original stations records file
# ONLY NEEDED TO COMBINE METEOROLOGICAL FILES

file3 = "H1-16_stations-2019-01au2021-05-30_Nicolet.csv"     # File from Jan 2019 to May 2021
st_rec1 = pd.read_csv(folder2 + file3, sep=';')              # Dataframe for the period

# %% Info-Climat files: New stations records file
folder3 = "C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Info/20220202-Meteo(2021)/"
file4 = "H1-14_stations-2021_J-D_Nicolet_TTP.csv"            # File from Jun 2021 to Dec 2021
st_rec2 = pd.read_csv(folder3 + file4, sep=';')              # Dataframe for the period
# st_records = st_rec2.copy()

# %% Combine first two files
st_rec_12 = pd.concat([st_rec1.set_index('Date'), st_rec2.set_index('Date')], axis=0, join="inner", ignore_index=False )
st_rec_12 = st_rec_12.reset_index()

# %% Info-Climat files: 2022 stations records file
# ONLY NEEDED TO COMBINE METEOROLOGICAL FILES
folder4 = "C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Info/20220620-Meteo(2022)/"
file5 = "H1-14_stations-2022_Nicolet_1.csv"                 # File from Jan 2022 to Jun 2022
st_rec3 = pd.read_csv(folder4 + file5, sep=';')              # Dataframe for the period

# %% Combine final two files
st_records = pd.concat([st_rec_12.set_index('Date'), st_rec3.set_index('Date')], axis=0, join="inner", ignore_index=False )
st_records = st_records.reset_index()

# %% Organize the meteorological data from Info-Climat

def kr_melcc(hyd_grid, hyd_gr_data, station_info, station_rec, meteo_var, coor_type):
    """
    Process meteorological data with the MELCC method to create grids.

    Parameters
    ----------
    hyd_grid : Dataframe
        From csv file containing names and coordinates for the grid
    hyd_gr_data : Dataframe
        From csv files containing meteorological variables of each grid station
    station_info : Dataframe
        From csv file containing names and coordinates for the stations.
    station_rec : Dataframe
        From csv file that contains meteorological records.
    meteo_var : String
        To define the type of variable to interpolate. Options: PT, TMax, Tmin
    coor_type : String
        Type of coordinates for kriging interpolations. Options: euclidean, geodesic

    Returns
    -------
    DataFrame
        Interpolated variable values to the defined grid.

    """
    print('Starting calculations for '+meteo_var)

    # Organizing Info-Climat file into a DataFrame for the variable
    if meteo_var == 'PT':
        base_1 = station_rec[(station_rec["Variable"] == 'PT')].drop(['Variable'], axis=1).set_index(['Date', 'ID'])
        base_2 = station_rec[(station_rec["Variable"] == 'PT_p')].drop(['Variable'], axis=1).set_index(['Date', 'ID'])
        base_df = base_1.combine_first(base_2)
        base_df = base_df.reset_index('ID')
    else:
        base_df = station_rec[(station_rec["Variable"] == meteo_var)].drop(['Variable'], axis=1).set_index('Date')

    # Organizing Info-Climat file into a dictionary
    sta_dic = dict.fromkeys(base_df['ID'].unique())  # Dictionary containing all stations
    sta_names = list(sta_dic.keys())  # Info-Climat station names for the current variable
    for sta in sta_names:
        sta_dic["{0}".format(sta)] = base_df[(base_df['ID'] == sta)].drop(['ID'], axis=1).stack(dropna=False)

    # % Functions to organize the data

    var_df = pd.DataFrame(columns=sta_names)  # Initializing variable dataframe for calculations
    # Organizing the variable into 3 hour blocks and assigning Hydrotel date format
    for sta in sta_names:
        vec = sta_dic[sta]  # Variable vector for each station
        if meteo_var == 'PT':
            var_xh = vec.groupby(np.arange(len(vec))//3).sum().to_frame()  # Series containing sums each 3 hours
        else:
            var_xh = vec.groupby(np.arange(len(vec))//3).mean().to_frame()  # Series containing average each 3 hours
        # Transform dates/hours to Hydrotel format
        ix_hr = list(vec.index)[0::3]  # Indexes from Date and time each 3 hours
        dtix = pd.to_datetime([i[0] for i in ix_hr]).strftime("%d/%m/%Y")  # Dates to Hydrotel format
        hrix = pd.to_datetime([i[1] for i in ix_hr]).strftime("%H")  # Hours to Hydrotel format
        h24 = ['24', '21', '18', '15', '12', '09', '06', '03', '00']  # Aux list to account for the 24h format
        for h in list(range(len(h24)-1)):
            hrix = [hr.replace(h24[h+1], h24[h]) for hr in hrix]  # Format hour replacing
        var_xh.index = pd.MultiIndex.from_arrays([dtix, hrix], names=['Date', 'Hour'])
        var_xh = var_xh[~var_xh.index.duplicated()]
        var_df[sta] = var_xh[0]  # Assign the Series to the variable Dataframe

    # % Coordinates transformation

    # Coordinates projection to the Lambert Conformal Conic
    # Origin = EPSG:4326 = WGS84
    # Destination = EPSG:32198 Quebec Lambert (Options: 32198 or 6622 or 3798)
    transformer = proj.Transformer.from_crs(4326, 32198)  # Object to transform the coordinates
    x_tr, y_tr = transformer.transform(np.array(station_info['Lat']), np.array(station_info['Lon']))
    # Dataframe for transformed coordinates of Info-Climat stations
    st_coord_pj = pd.DataFrame(np.array([y_tr, x_tr]), index=['Lat/Y', 'Lon/X'], columns=station_info.index)

    # % Normalize temperature for altitude

    var_dfn = var_df.copy()  # Normalized variable (PT will not be modified)

    if meteo_var != 'PT':
        # By every 100 m of altitude, temperatures are reduced in 0.5 C
        t_norm = station_info['Alt'] / 100 * 0.5
        for sta in var_df.columns:
            var_dfn[sta] = var_df[sta].apply(lambda x: x - t_norm.loc[sta])
    var_dfn.index.name = meteo_var

    # % Kriging interpolation

    # Projecting grid coordinates to euclidean system
    gr_names = np.array(hyd_grid.index)
    x_gr, y_gr = transformer.transform(np.array(hyd_grid['Lat']),
                                       np.array(hyd_grid['Lon']))  # Coordinate transformation
    gr_coord_pj = pd.DataFrame(np.array([y_gr, x_gr]),
                               index=['Lat/Y', 'Lon/X'], columns=gr_names)  # Dataframe for transformed coordinates

    # % Build new variable grid
    var_int_grid = Kriging.kint(var_dfn, st_coord_pj, gr_coord_pj, meteo_var, coor_type)

    # % Variable adjustments
    var_new_grid = var_int_grid.copy()  # Initializing / If no condition is met

    # By every 100 m of altitude, temperatures are augmented in 0.5 C
    if meteo_var != 'PT':
        t_corr = hyd_grid['Alt'] / 100 * 0.5
        for stn in hyd_grid.index:
            var_new_grid[stn] = var_int_grid[stn].apply(lambda x: x + t_corr.loc[stn])

    # Assigning 0 to all negative precipitations
    elif (var_int_grid < 0).any(None):
        var_new_grid = var_int_grid.mask(var_int_grid.iloc[:, :] < 0, 0, inplace=False)

    return var_new_grid, var_dfn

    # %% Performance results

def kr_melcc_perf(hyd_grid, hyd_gr_data, meteo_var, var_new) :
    # Extracting variable from Hydrotel grid data file
    hyd_df = pd.DataFrame(columns=hyd_grid.index, index=hyd_gr_data.index)  # Initializing Hydrotel variable dataframe

    for hsta in hyd_grid.index:
        hyd_df[hsta] = hyd_gr_data[hsta + '-' + meteo_var]

    # % Slice dataframes based on overlapping dates
    st_hyd_ix = hyd_df.index.get_loc(var_new.index[0])  # Index from where to slice Hydrotel DF
    hyd_df_sl = hyd_df.iloc[st_hyd_ix::, :]  # Sliced hydrotel DF

    en_var_ix = var_new.index.get_loc(hyd_df.index[-1])+1  # Index up to where to slice new DF
    var_nw_sl = var_new.iloc[0:en_var_ix, :]  # Sliced new variable DF

    # % Build a performance dataframe
    def performance(obs, pre, var):
        perf_df = pd.DataFrame(index=obs.columns, columns=['ME', 'MAE', 'MSE', 'RMSE'])

        for sta in obs.columns:
            perf_df.loc[sta]['ME'] = (obs[sta]-pre[sta]).mean()
            perf_df.loc[sta]['MAE'] = mae(obs[sta], pre[sta])
            perf_df.loc[sta]['MSE'] = mse(obs[sta], pre[sta], squared=False)
            perf_df.loc[sta]['RMSE'] = mse(obs[sta], pre[sta], squared=True)
        perf_df.index.name = var
        return perf_df

    vr_perf = performance(hyd_df_sl, var_nw_sl, meteo_var)  # Performance results for variable

    # % Plot results

    # Plotting results of new interpolation grid
    def comp_plot(interp, ref, pltvar, point):
        fig = pd.concat([interp[point], ref[point]], axis=1)
        fig.columns = ['Interpolation', 'Hydrotel']
        fig.plot(y=['Interpolation', 'Hydrotel'], use_index=True, kind='line', rot=90, title=pltvar+'-'+point)
        plt.show()

    comp_plot(var_nw_sl, hyd_df_sl, meteo_var, '719_456')

    return vr_perf

#%% Run precipitation interpolation
pt_new_grid, pt_sta_grid = kr_melcc(hyd_coord, hyd_gr_df, st_info, st_records, 'PT', 'euclidean')
#pt_perf = kr_melcc_perf(hyd_coord, hyd_gr_df, 'PT', pt_new_grid)
#print(pt_perf)

#%% Run Tmin interpolation
tmin_new_grid, tmin_sta_grid = kr_melcc(hyd_coord, hyd_gr_df, st_info, st_records, 'Tmin', 'euclidean')
#tmin_perf = kr_melcc_perf(hyd_coord, hyd_gr_df, 'Tmin', tmin_new_grid)
#print(tmin_perf)

#%% Run TMax interpolation
tmax_new_grid, tmax_sta_grid = kr_melcc(hyd_coord, hyd_gr_df, st_info, st_records, 'TMax', 'euclidean')
tmax_new_grid[tmax_new_grid<=tmin_new_grid] = tmin_new_grid     # Correction for tmax <= tmin
#tmax_perf = kr_melcc_perf(hyd_coord, hyd_gr_df, 'TMax', tmax_new_grid)
#print(tmax_perf)

# %% Append results to Hydrotel meteorological files

st_pt_ix = pt_new_grid.index.get_loc(hyd_gr_df.index[-1])+1  # Starting index from where to append precipitation
st_tn_ix = tmin_new_grid.index.get_loc(hyd_gr_df.index[-1])+1  # Starting index from where to append tmin
st_tx_ix = tmax_new_grid.index.get_loc(hyd_gr_df.index[-1])+1  # Starting index from where to append tmax

frmt = '%+10s %+2s %+5s %+5s %+4s'  # Format of each line for the text files

folder_w = "C:/Users/reds2401/Documents/HYDROTEL/Hydrotel_030101/meteo_krigage/"

for hgsta in hyd_coord.index:
    # Organize TMAx, Tmin and PT into a df for each station
    df_app = pd.concat([tmax_new_grid[hgsta][st_tx_ix::], tmin_new_grid[hgsta][st_tn_ix::],
                        pt_new_grid[hgsta][st_pt_ix::]], axis=1).astype(float).round(1).reset_index()
    meteo_file = open(folder_w + hgsta + '.met', 'a')  # Open the file in 'append' mode
    np.savetxt(meteo_file, df_app.values, fmt=frmt)  # Writting the file
    meteo_file.close()
