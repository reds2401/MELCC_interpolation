# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:21:21 2022

@author: reds2401
"""

# %% Libraries
import pandas as pd


# %% Info-Climat files

# Import file with stations coordinates
folder1 = "C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Info/20210603-Meteo(2019-2021)/"
folder2 = "C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Info/20220202-Meteo(2021)/"
# folder2 = "C:/Users/sergi/Downloads/OneDrive_1_7-7-2021/"
description_file = "Stations_Nicolet.csv"

# Dataframe with station coordinates
st_info = pd.read_csv(folder1 + description_file, sep='\s+').set_index('NO_STATION').drop(['NOM_STATION'], axis=1)
st_info.columns = ['Lat', 'Lon', 'Alt']

# Import Info-Climat meteorological data file
file1 = "H1-16_stations-2019-01au2021-05_Nicolet.csv"
st_records1 = pd.read_csv(folder1 + file1, sep=';')  # Dataframe with all stations and variables

file2 = "H1-14_stations-2021_Nicolet.csv"
st_records2 = pd.read_csv(folder2 + file2, sep=';')  # Dataframe with all stations and variables

st_records_full = pd.concat([st_records1.set_index('Date'), st_records2.set_index('Date')], axis=0, join="inner", ignore_index=False )
# st_records_full = st_records_full.drop_duplicates().reset_index()