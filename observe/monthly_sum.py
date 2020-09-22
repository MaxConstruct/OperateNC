import xarray as xr
import numpy as np
import pandas as pd
import util.netcdf_util as ut
from pathlib import Path

daily_p = Path(r'H:\Observation\Cleaned Data\daily 1998-2014')
monthly_p = Path(r'H:\Observation\Cleaned Data\monthly 1998-2014')
#%%
pr_ds = [i for i in daily_p.iterdir()]
#%%
for i in pr_ds:
    da = xr.open_dataarray(i)
    n_ds = da.resample(time='M').sum(skipna=False)
    n_ds.attrs = {
        'long_name': 'Total precipitation',
        'units': 'mm/month'
    }
    n_ds.to_netcdf(monthly_p / (i.name.split('_')[0] + '_pr_monthly_1998_2014.nc'))

#%%
da = xr.open_dataarray(pr_ds[5])
n_ds = da.resample(time='M').sum(skipna=False)
n_ds.attrs = {
    'long_name': 'Total precipitation',
    'units': 'mm/month'
}
n_ds.to_netcdf(monthly_p / (pr_ds[5].name.split('_')[0] + '_pr_monthly_1998_2014.nc'))
