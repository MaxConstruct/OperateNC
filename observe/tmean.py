import xarray as xr
import numpy as np
import pandas as pd
from observe import observe_path as op
import util.netcdf_util as ut
from pathlib import Path

# %%
clean_dir = Path(r'H:\Observation\Cleaned Data\tmean')
daily = clean_dir / 'daily 1998-2014'
monthly = clean_dir / 'monthly 1998-2014'
mean = clean_dir / 'mean annual'
#%%
daily.mkdir()
monthly.mkdir()
mean.mkdir()
# %%
era_p = [
    r'H:\Observation\Cleaned Data\tmax\era_tasmax_daily_1979_2018.nc',
    r'H:\Observation\Cleaned Data\tmin\era_tasmin_daily_1979_2018.nc'
]

jra_p = [
    r'H:\Observation\Cleaned Data\tmax\jra55_tasmax_daily_1998_2018.nc',
    r'H:\Observation\Cleaned Data\tmin\jra55_tasmin_daily_1998_2018.nc'
]
#%%
ds1 = xr.open_dataarray(jra_p[0])
ds2 = xr.open_dataarray(jra_p[1])
ds_mean = 0.5 * (ds1 + ds2)
ds_mean = ds_mean.rename('tmean')
#%%
ds_mean.attrs = {
    'units': 'degree centigrade',
    'long_name': 'Near-surface mean temperature',
    'note': "The daily 'mean' temperature is the midpoint (median) between the daily minimum and maximum temperatures."
}
#%%
ds_mean.to_netcdf(clean_dir / 'jra55_tmean_daily_1998_2018.nc')
#%%

cru_p = r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.tmp.dat.nc'
aph_p = r'H:\Observation\Raw Data (SEA)\[SEA] Aphrodite 1901 & 1808\APHRO_MA_TAVE_025deg_V1808_1961-2015_Temp.nc'
cru = xr.open_dataset(cru_p)
aph = xr.open_dataset(aph_p)