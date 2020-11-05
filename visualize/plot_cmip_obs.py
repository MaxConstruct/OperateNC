# %%
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import util.netcdf_util as ut
import geopandas as gs
from pathlib import Path
from os import startfile as st
import matplotlib.pyplot as plt
import util.preprocess as pre
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def crop_land_only(base, dataset):
    return dataset.where_greater(base.notnull())


def sum_monthly_to_year(ds):
    return ds.resample(time='AS').sum(skipna=False)


def mean_monthly_to_year(ds):
    return ds.resample(time='AS').mean(skipna=False)


def mean_lat_lon_all_time(ds):
    return np.array([ds.isel(time=i).mean(dim=['lat', 'lon'], skipna=True) for i in range(len(ds))])


def ds_name(full_name) -> str:
    return full_name.split('_')[4].upper()


def get_path(path: Path):
    return sorted(list(path.iterdir()))


def add_name(ds: xr.DataArray, name: str):
    return ds.assign_coords(id=name)


# %%
obs_tmean = xr.open_dataarray(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014\cru_tmean_monthly_1998_2014.nc')
tmean_hist_path = Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_tmean_hist_1998_2014_noleap')
obs_tmean_mean = obs_tmean.mean(dim='time')

tmean_ds_list = [
    crop_land_only(base=obs_tmean_mean,
                   dataset=add_name(xr.open_dataarray(p), ds_name(p.name)).mean(dim='time')
                   )
    for p in get_path(tmean_hist_path)
]
tmean_ds_list = [ds.drop_vars('height') for ds in tmean_ds_list]
obs_tmean_mean = obs_tmean_mean.assign_coords(id='CRU')
tmean_ds_list = [obs_tmean_mean] + tmean_ds_list

for i in range(len(tmean_ds_list)):
    tmean_ds_list[i] = tmean_ds_list[i].assign_coords(
        id=f'({chr(ord("a") + i)}) ' + str(tmean_ds_list[i].coords['id'].values))
tmean_ds = xr.concat(tmean_ds_list, dim='id')
#%%

obs_pr = xr.open_dataarray(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc')
pr_hist_path = Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap')
#%%
obs_pr_mean = obs_pr.resample(time='AS').sum(skipna=False).mean(dim='time', skipna=False)
#%%
pr_ds_list = [
    crop_land_only(base=obs_pr_mean,
                   dataset=add_name(xr.open_dataarray(p) * 86400, ds_name(p.name)).resample(time='AS').sum(skipna=False).mean(dim='time', skipna=False)
                   )
    for p in get_path(pr_hist_path)
]
#%%
obs_pr_mean = obs_pr_mean.assign_coords(id='CRU')
pr_ds_list = [obs_pr_mean] + pr_ds_list
for i in range(len(pr_ds_list)):
    pr_ds_list[i] = pr_ds_list[i].assign_coords(
        id=f'({chr(ord("a") + i)}) ' + str(pr_ds_list[i].coords['id'].values)
    )
pr_ds = xr.concat(pr_ds_list, dim='id')
# %%
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
#%%
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 18

#%%
def mf_plot(ds, levels=None):
    size = 23
    p = ds.plot(col='month', col_wrap=4, cmap='jet',
                transform=ccrs.PlateCarree(),
                aspect=1,
                figsize=[2*x for x in plt.rcParams["figure.figsize"]],
                # levels=levels,
                cbar_kwargs={
                    'spacing': 'proportional',
                    'orientation': 'horizontal',
                    'shrink': 1,
                    'label': '',
                    'aspect': 40,
                    'anchor': (0.5, 2.0),
                },
                subplot_kws={
                    'projection': ccrs.PlateCarree()
                },
                )
    for i, ax in enumerate(p.axes.flat):
        ax.coastlines()
        ax.add_feature(country_borders, edgecolor='darkgray')
        ax.set_extent([92.5, 142.5, -12.5, 24.5], crs=ccrs.PlateCarree())

    plt.show()
    plt.clf()
    plt.close()
mf_plot(new_ds)
#%%
mf_plot(tmean_ds)

#%%

pr_hist_path = Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap')

#%%

get_path(pr_hist_path)[11]