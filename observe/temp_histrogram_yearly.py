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

cpc_base_land = xr.open_dataarray(r'H:\Observation\Cleaned Data\pr\mean annual 1998-2014\cpc_pr_annual_1998_2014.nc')
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
to_cut_name_temp = [i.upper() for i in ['era-inter', 'jra55']]
to_interp_name_temp = [i.upper() for i in ['SA-OBS', 'aphrodite']]
temp_aphrodite_base_land = xr.open_dataarray(
    r'H:\Observation\Cleaned Data\tmean\mean annual 1998-2014\aphrodite_tmean_annual_1998_2014.nc')
temp_path_root = Path(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014')
temp_path = sorted(list(temp_path_root.iterdir()))
temp_ds = [xr.open_dataarray(i) for i in temp_path]

temp_ds.insert(0, temp_ds.pop(-1))
temp_path.insert(0, temp_path.pop(-1))


# %%
def crop_land_only(base, dataset):
    return dataset.where_greater(base.notnull())


def sum_monthly_to_year(ds):
    return ds.resample(time='A').sum(skipna=False)


def mean_monthly_to_year(ds):
    return ds.resample(time='A').mean(skipna=False)


def mean_lat_lon_all_time(ds):
    return np.array([ds.isel(time=i).mean(dim=['lat', 'lon'], skipna=True) for i in range(17)])


def ds_name(paths, i):
    return paths[i].name.split('_')[0].upper()


# %%
# For mean temperature
mf_ds = []
for i, dset in enumerate(temp_ds):
    print(i, '\r', flush=True, end='')
    temp = dset.copy()
    dataset_name = ds_name(temp_path, i)
    if dataset_name in to_interp_name_temp:
        temp = pre.regrid_sea(temp)
    # if dataset_name in to_cut_name_temp:
    #     print(dataset_name)
    temp = crop_land_only(cpc_base_land, temp)

    mf_ds.append(temp.assign_coords(id=dataset_name))
print('Done.')
# %%
# for mean Temperature
yearly = [mean_monthly_to_year(i) for i in mf_ds]
mean17 = [i.mean(dim='time', skipna=True) for i in yearly]
mean17_single = np.array([i.mean(dim=['lat', 'lon'], skipna=True) for i in mean17])
mean_yearly = np.array([mean_lat_lon_all_time(i) for i in yearly])


# %%
def mf_plot(_mf_ds, paths, level, time_enable=False, time=0):
    fig, axes = plt.subplots(ncols=3,
                             nrows=2,
                             figsize=(16, 9),
                             subplot_kw={
                                 # 'projection': ccrs.PlateCarree()
                             },
                             tight_layout=False)
    faxes = axes.flatten()
    fig.delaxes(faxes[-1])
    # fig.delaxes(faxes[-2])
    for i, ax in enumerate(faxes[:len(_mf_ds)]):

        if time_enable:
            dat = _mf_ds[i].isel(time=time)
        else:
            dat = _mf_ds[i]

        name = ds_name(paths, i)

        if name == 'SA-OBS'.upper():
            cpc_base_land.plot(ax=ax,
                               levels=[0],
                               cmap='Set2_r',
                               label=None,
                               add_colorbar=False
                               )
        dat.plot(ax=ax,
                 levels=level,
                 cmap='jet',
                 )

        # ax.coastlines()
        ax.set_title(f'({chr(i + 97)}) {name}')

    # plt.savefig(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img\Mean temp annual.png')
    plt.show()
    plt.clf()
    plt.close()


mf_plot(mean17, temp_path,  np.arange(5, 31, 0.01))

# %%
for i, m in enumerate(mean_yearly):
    print(ds_name(temp_path, i), '\t', '\t'.join([str(j) for j in m]))
# print('MEAN')
# for i, m in enumerate(mean17_single):
#     print(m)

# %%
ann_p = Path(r'H:\Observation\Cleaned Data\tmean\mean annual 1998-2014')
ls = []
paths = sorted(list(ann_p.iterdir()))
paths.insert(0, paths.pop(-1))
#%%
for i, p in enumerate(paths):
    print(p.name)
    ds = xr.open_dataarray(p)
    name = ds_name(paths, i)
    if name in to_interp_name_temp:
        temp = pre.regrid_sea(ds)
    # if dataset_name in to_cut_name_temp:
    #     print(dataset_name)
    temp = crop_land_only(cpc_base_land, ds)
    ls.append(ds.mean(dim=['lat', 'lon']).values)
mean_ann = np.array(ls)
print(*mean_ann, sep='\n')

#%%
t_ls = [i.mean(dim=['lat', 'lon'], skipna=True) for i in mf_ds]
re = np.array(t_ls)
for i, m in enumerate(re):
    print(ds_name(temp_path, i), *m, sep='\t')
