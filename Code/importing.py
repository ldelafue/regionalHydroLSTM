"""
This code create a common database depending of the country of the original dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
import os
import datetime




def importing(catchment:str):
    """Load variables (PP, PET, Q) from US datset
    Parameters
    ----------
    catchment : string
        ID of the basin

    Returns
    -------
    pd.DataFrame
        PP: Precipitation in mm/day
        PET: Potential evapotranspiration in mm/day (by Hargreaves & Samani equation)
        Q: Streamflow in mm/day

    """

    # Checking the format of the catchment name
    if isinstance(catchment, str):
        if len(catchment) == len(str(int(catchment))) and int(catchment)<10000000:
            catchment = '0' + catchment
    else:
        if catchment<10000000:
            catchment = '0' + str(catchment)
        else:
            catchment = str(catchment)

    #Definition of the path
    path = '/Users/mac_laptop/CAMELS/data/' #'/groups/hoshin/ldelafue/PhD/CAMELS/data/' #'/Users/mac_laptop/CAMELS/data/' #/Users/luis/CAMELS/data/camels_attributes_v2.0/camels_name.txt
    path = Path(path)

    # Loading data
    HUC_number_path = path / 'camels_attributes_v2.0' / 'camels_name.txt' # os.getcwd() / 
    col_names = ['gauge_id','huc_02','gauge_name']
    HUC_number = pd.read_csv(HUC_number_path, sep=';', header=0, names=col_names)
    HUC_number.index = HUC_number.gauge_id.values        

    HUC = HUC_number.huc_02[int(catchment)]
    if HUC<10:
        HUC = '0' + str(HUC)
    else:
        HUC = str(HUC)
  
    forcing_file = catchment + '_lump_maurer_forcing_leap.txt'
    forcing_path = path / 'basin_mean_forcing' / 'maurer_extended' / HUC # I have to generalize that
    forcing_path = os.getcwd() / forcing_path / forcing_file
    
    Q_file = catchment + '_streamflow_qc.txt'
    Q_path = path / 'usgs_streamflow' / HUC # I have to generalize that
    Q_path = os.getcwd() / Q_path / Q_file

    Topo_file = 'camels_topo.txt'
    Topo_path = path / 'camels_attributes_v2.0'
    Topo_path = os.getcwd() / Topo_path / Topo_file



    #Reading the files

    col_names = ['gauge_id','gauge_lat','gauge_lon','elev_mean','slope_mean','area_gages2','area_geospa_fabric']
    Topo_df = pd.read_csv(Topo_path, sep=';', header=0, names=col_names)
    Topo_df.index = Topo_df.gauge_id
    lat = Topo_df.gauge_lat[int(catchment)]*2*np.pi/360

    forcing_df = pd.read_csv(forcing_path, sep='\s+', header=3)
    dates = (forcing_df.Year.map(str) + "/" + forcing_df.Mnth.map(str) + "/" + forcing_df.Day.map(str))
    jul = pd.to_datetime(forcing_df.Year.map(str)+ "/01/01" , format="%Y/%m/%d")
    forcing_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    forcing_df = forcing_df.drop(['Year','Mnth','Day','Hr','dayl(s)'], axis=1)
    forcing_df['basin'] = int(catchment)


    forcing_df['PET'] = 0.408*0.0023*(forcing_df['tmax(C)'] - forcing_df['tmin(C)'])**0.5*(0.5*forcing_df['tmax(C)'] + 0.5*forcing_df['tmin(C)'] + 17.8)
    forcing_df['julian'] = pd.DatetimeIndex(forcing_df.index).to_julian_date() - pd.DatetimeIndex(jul).to_julian_date() + 1

    forcing_df['gamma'] = 0.4093*np.sin(2*np.pi*forcing_df.julian/365 - 1.405)
    forcing_df['hs'] = np.arccos(-np.tan(lat)*np.tan(forcing_df.gamma))
    forcing_df['PET'] = 3.7595*10*(forcing_df.hs*np.sin(lat)*np.sin(forcing_df.gamma)+np.cos(lat)*np.cos(forcing_df.gamma)*np.sin(forcing_df.hs))*forcing_df.PET

    forcing_df['basin'] = int(catchment)

    PP_df = forcing_df[['basin', 'prcp(mm/day)']].rename(columns={"prcp(mm/day)": 'PP'})
    PET_df = forcing_df[['basin', 'PET']]
    PET_df.loc[PET_df.PET<0,'PET'] = 0

    with open(forcing_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'Q_obs', 'flag']
    Q_df = pd.read_csv(Q_path, sep='\s+', header=None, names=col_names)
    dates = (Q_df.Year.map(str) + "/" + Q_df.Mnth.map(str) + "/" + Q_df.Day.map(str))
    Q_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    Q_df = Q_df.drop(['Year', 'Mnth', 'Day', 'flag'], axis=1)
    Q_df.loc[Q_df.Q_obs < 0, 'Q_obs'] = 0
    Q_df.Q_obs = 28316846.592 * Q_df.Q_obs * 86400 / (area * 10 ** 6)


    return PP_df, PET_df, Q_df

