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


# %%
# prs = [i for i in clean_dir.iterdir()]
# pr_attr = {'units': 'mm', 'long_name': 'Total precipitation'}
# cru = xr.open_dataset(prs[6]).rename({'pre': 'pr'})
# d = cru.resample(time='1D').mean('time').groupby('time')
# dates, datasets = zip(*d)
# filenames = [clean_dir / 'cru_days' / (pd.to_datetime(date).strftime('%Y.%m.%d') + '.nc') for date in dates]
# xr.save_mfdataset(datasets, filenames)
# for p in prs:
#     name = p.name.split('_')[0]
#     with xr.open_dataset(p) as ds:
#         print(name)
#         n_ds = ut.select_year(ds, 1998, 2014)
#         n_ds.to_netcdf(clean_dir / '1998-2014' / (name + '_pr_1998_2014.nc'))


# %%

#
# def set_attr(ds, attr):
#     ds.attrs = attr
#     return ds
#
#
# def make_resample(ds, time, method):
#     if method == 'mean':
#         return ds.resample(time=time).mean()
#     elif method == 'sum':
#         return ds.resample(time=time).sum()
#
#     raise ValueError(f'Unsupported method: {method}')
#
#
# def get_dataarray(ds, var):
#     if isinstance(ds, xr.Dataset):
#         return ds[var]
#     elif isinstance(ds, xr.DataArray):
#         return ds
#
# def make_structure(ds, var, resample_method, conclude_method='', rename=None, attrs={}):
#     if rename is not None:
#         ds = ds.rename(rename)
#         var_ds = get_dataarray(ds, rename[var])
#     else:
#         var_ds = get_dataarray(ds, var)
#
#     re = [('init', var_ds)]
#
#     for method in resample_method:
#         re.append((method[0], set_attr(make_resample(re[-1][-1], time=method[0], method=method[1]), attrs)))
#
#     if conclude_method == 'mean':
#         re.append(('con', set_attr(re[-1][-1].mean(dim='time'), attrs)))
#     elif conclude_method == 'sum':
#         re.append(('con', set_attr(re[-1][-1].sum(dim='time'), attrs)))
#     else:
#         raise ValueError(f'Unsupported method {conclude_method}')
#
#     return {i[0]: i[1] for i in re}
#

# %%
tmean = [
    r'H:\Observation\Cleaned Data\tmean\era_tmean_daily_1979_2018.nc',
    r'H:\Observation\Cleaned Data\tmean\jra55_tmean_daily_1998_2018.nc',
    r'H:\Observation\Cleaned Data\tmean\aph_tmean_daily_1961_2015.nc',
    r'H:\Observation\Cleaned Data\tmean\cru_tmean_monthly_1901_2019.nc'
]

#%%
# re = make_structure(era,
#                     'mn2t',
#                     resample_method=[('M', 'mean')],
#                     conclude_method='mean',
#                     rename={
#                         'mn2t': 'tasmin',
#                     },
#                     attrs=era.mn2t.attrs
#                     # {
#                     #     'units': 'degree centigrade',
#                     #     'long_name': 'Minimum temperature at 2 metres'}
#                     )

#%%
def get_year_bounds(ds):
    return str(ds.time.dt.year[0].values), str(ds.time.dt.year[-1].values)
#%%
path = Path(tmean[3])
ds = xr.open_dataset(path)
#%%
name = path.name.split('_')[0]
attr1 = {
    'units': 'degree centigrade',
    'long_name': 'Minimum temperature at 2 metres'
}
print(name)
#%%
ds_tmin = ds['tmean']
# ds_tmin = ds_tmin.rename('tasmin')
# ds_tmin = ds_tmin.rename({'initial_time0_hours': 'time'})
# y1, y2 = get_year_bounds(ds_tmin)
# ds_tmin.to_netcdf(clean_dir / f'_tasmin_daily_{y1}_{y2}.nc')
print(ds_tmin)
#%%
# 'JRA55 h to 1D'
# ds_d = ds_tmin.resample(time='1D').mean(keep_attrs=True)
# y1, y2 = get_year_bounds(ds_d)
# ds_d.attrs = attr1
# ds_d.to_netcdf(clean_dir / f'{name}_tasmin_daily_{y1}_{y2}.nc')
#%%
ds_d = ut.select_year(ds_tmin, 1998, 2014)
# ds_d.to_netcdf(daily / f'{name}_tmean_daily_{1998}_{2014}.nc')
#%%
ds_m = ut.select_year(ds_tmin, 1998, 2014).resample(time='M').mean(keep_attrs=True)
# ds_m.to_netcdf(monthly / f'{name}_tmean_monthly_{1998}_{2014}.nc')
#%%
ds_mean = ds_m.mean(dim='time', keep_attrs=True)
ds_mean.to_netcdf(mean / f'{name}_tmean_annual_{1998}_{2014}.nc')

