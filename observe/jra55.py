# %%
import xarray as xr
import numpy as np
import pandas as pd
from observe import observe_path as op
import util.netcdf_util as ut
from pathlib import Path

clean_dir = Path(r'H:\Observation\Cleaned Data')
daily = 'daily 1998-2014'
monthly = 'monthly 1998-2014'
mean = 'mean annual 1998-2014'
raw_p = [
    r'H:\Observation\Raw Data (SEA)\[SEA] JRA-55\minmax_surf.016_tmax.reg_tl319.1998010100_2018123121.nc',
    r'H:\Observation\Raw Data (SEA)\[SEA] JRA-55\minmax_surf.016_tmin.reg_tl319.1998010100_2018123121.nc'
]

#%%
def make_summary(ds_path, ds_name, ds_varname, save_var, attrs):
    if save_var not in ['tasmax', 'tasmin']:
        raise ValueError('Unsupported ' + ds_varname)
    ds = xr.open_dataset(ds_path)
    ds_var = ds[ds_varname]
    ds_var = ds_var.rename({'initial_time0_hours': 'time'})
    ds_var = ds_var.rename(save_var)
    ds_var.attrs = attrs
    if save_var == 'tasmax':
        ds_day = ds_var.resample(time='1D').max(keep_attrs=True)
    else:
        ds_day = ds_var.resample(time='1D').min(keep_attrs=True)
    make_sum2(ds_day, ds_name, save_var)
    return ds_day
#%%

def make_sum2(ds_day, ds_name, save_var):
    ds_day_cut = ut.select_year(ds_day, 1998, 2014)
    ds_monthly = ds_day_cut.resample(time='M').mean(keep_attrs=True)
    ds_mean = ds_monthly.mean(dim='time', keep_attrs=True)

    ds_day.to_netcdf(clean_dir / save_var / f'{ds_name}_{save_var}_daily_1998_2018.nc')
    ds_day_cut.to_netcdf(clean_dir / save_var / daily / f'{ds_name}_{save_var}_daily_1998_2014.nc')
    ds_monthly.to_netcdf(clean_dir / save_var / monthly / f'{ds_name}_{save_var}_monthly_1998_2014.nc')
    ds_mean.to_netcdf(clean_dir / save_var / mean / f'{ds_name}_{save_var}_annual_1998_2014.nc')

# %%
jmax = make_summary(
    ds_path=raw_p[0],
    ds_name='jra55',
    ds_varname='TMAX_GDS4_HTGL',
    save_var='tasmax',
    attrs={
        'units': 'degree centigrade',
        'long_name': 'Near-surface maximum temperature'
    })
jmin = make_summary(
    ds_path=raw_p[1],
    ds_name='jra55',
    ds_varname='TMIN_GDS4_HTGL',
    save_var='tasmin',
    attrs={
        'units': 'degree centigrade',
        'long_name': 'Near-surface minimum temperature'
    })
jmean = (jmax + jmin)*0.5
make_sum2(ds_day=jmean, ds_name='jra55', save_var='tmean')
var_name = [
    'TMAX_GDS4_HTGL',
    'TMIN_GDS4_HTGL'
]


