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

# %%
cpc_base_land = xr.open_dataarray(r'H:\Observation\Cleaned Data\pr\mean annual 1998-2014\cpc_pr_annual_1998_2014.nc')
#%%
to_cut_name = [i.upper() for i in
               ['era-inter', 'gpcc', 'gpcp', 'jra55', 'trmm', 'cmorph']
               ]
to_interp_name = [i.upper() for i in
                  ['CMORPH', 'CHIRPS', 'GPCC', ]
                  ]

# %%
pr_path_root = Path(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014')
pr_path = sorted(list(pr_path_root.iterdir()))
pr_ds = [xr.open_dataarray(i) for i in pr_path]

#%%
temp_path_root = Path(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014')
temp_path = sorted(list(pr_path_root.iterdir()))
temp_ds = [xr.open_dataarray(i) for i in pr_path]

# %%
def crop_land_only(base, dataset):
    return dataset.where(base.notnull())


# %%
mf_ds = []
for i, dset in enumerate(pr_ds):
    print(i, '\r', flush=True, end='')
    temp = dset.copy()
    dataset_name = pr_path[i].name.split('_')[0].upper()
    if dataset_name in to_interp_name:
        temp = pre.regrid_sea(temp)
    if dataset_name in to_cut_name:
        temp = crop_land_only(cpc_base_land, temp)
    mf_ds.append(temp.assign_coords(id=dataset_name))
print('Done.')


# mf_ds = xr.concat(mf_arr, dim='id')

# %%
def sum_monthly_to_year(ds):
    return ds.resample(time='A').sum(skipna=False)


# %%
def mean_lat_lon_all_time(ds):
    return np.array([ds.isel(time=i).mean(dim=['lat', 'lon'], skipna=True) for i in range(17)])


# %%
yearly = [sum_monthly_to_year(i) for i in mf_ds]

#%%
def mf_plot(_mf_ds, time=0):
    fig, axes = plt.subplots(ncols=3,
                             nrows=4,
                             figsize=(14, 14),
                             subplot_kw={
                                 # 'projection': ccrs.PlateCarree()
                             },
                             tight_layout=False)
    faxes = axes.flatten()
    fig.delaxes(faxes[-1])
    fig.delaxes(faxes[-2])
    for i, ax in enumerate(faxes[:len(_mf_ds)]):
        _mf_ds[i].isel(time=time).plot(ax=ax,
                                       # levels=np.arange(10, 32),
                                       levels=np.arange(0, 6000),
                                       cmap='jet',
                                       label=None)
        # ax.coastlines()
        ax.set_title(pr_path[i].name.split('_')[0].upper())
    plt.savefig(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img\Mean annual.png')
    plt.show()
    plt.clf()
    plt.close()


mf_plot(yearly)
# mf_plot(mf_ds)
# %%


# %%
path_plot = Path(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img\monthly')
for i in range(len(mf_ds)):
    p = mf_ds[i].isel(time=slice(0, 12)).plot(
        x="lon",
        y="lat",
        col="time",
        col_wrap=4,
        add_colorbar=False,
    )
    p.fig.suptitle(pr_path[i].name)
    plt.savefig(path_plot / f'{pr_path[i].name}.png')
    plt.show()
    plt.clf()
    plt.close()
# %%
for i in range(len(yearly)):
    p = yearly[i].plot(
        x="lon",
        y="lat",
        col="time",
        col_wrap=4,
        add_colorbar=False,
    )
    p.fig.suptitle(pr_path[i].name)
    # plt.savefig(path_plot / f'{pr_path[i].name}.png')
    plt.show()
    plt.clf()
    plt.close()
# %%
mean_yearly = np.array([mean_lat_lon_all_time(i) for i in yearly])

# %%
pr_ann = Path(r'H:\Observation\Cleaned Data\pr\mean annual 1998-2014')
ls = []
for i, p in enumerate(sorted(list(pr_ann.iterdir()))):
    print(p.name)
    ds = xr.open_dataarray(p)
    if i >= 3:
        ds = crop_land_only(cpc_base_land, ds)
    ls.append(ds.mean(dim=['lat', 'lon']).values)
mean_ann = np.array(ls)
print(mean_ann)
