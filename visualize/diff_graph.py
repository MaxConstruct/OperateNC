# %%
from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np
import util.netcdf_util as ut
import matplotlib.pyplot as plt

# # cru = xr.load_dataarray(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014\cru_tmean_monthly_1998_2014.nc')
# cru = xr.load_dataarray(r'H:\Observation\Cleaned Data\pr\mean annual 1998-2014\cru_pr_annual_1998_2014.nc')
# cru_mean = cru.mean()
#
# ssp245_p = ut.lsdir(Path(r'H:\CMIP6 - Biased\tmean\ssp245'))
# ssp585_p = ut.lsdir(Path(r'H:\CMIP6 - Biased\tmean\ssp585'))
#
# ssp245_pr = ut.lsdir(r'H:\CMIP6 - Biased\pr_gamma\nc\ssp245')
# ssp585_pr = ut.lsdir(r'H:\CMIP6 - Biased\pr_gamma\nc\ssp585')
#
# ssp245 = ut.get_mfds(ssp245_pr)
# ssp585 = ut.get_mfds(ssp585_pr)
#%%
hist_pr = ut.lsdir(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap')
ssp245_pr = ut.lsdir(r'H:\CMIP6 - SEA\Cleaned\ssp245\decode_cmip_pr_ssp245_2015_2100_noleap')
ssp585_pr = ut.lsdir(r'H:\CMIP6 - SEA\Cleaned\ssp585\decode_cmip_pr_ssp585_2015_2100_noleap')

# hist = ut.get_mfds(hist_pr)
# ssp245 = ut.get_mfds(ssp245_pr)
# ssp585 = ut.get_mfds(ssp585_pr)
#%%
mf = []
for i, p in enumerate(hist_pr):
    mf.append(xr.load_dataarray(p).resample(time='MS').sum().assign_coords(id=i, time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')))
hist = xr.concat(mf, dim='id')

mf = []
for i, p in enumerate(ssp245_pr):
    mf.append(xr.load_dataarray(p).resample(time='MS').sum().assign_coords(id=i, time=pd.date_range('2015-01-01', '2100-12-01', freq='MS')))
ssp245 = xr.concat(mf, dim='id')

mf = []
for i, p in enumerate(ssp585_pr):
    mf.append(ut.select_year(xr.load_dataarray(p).resample(time='MS').sum(), 2015, 2100).assign_coords(id=i, time=pd.date_range('2015-01-01', '2100-12-01', freq='MS')))
ssp585 = xr.concat(mf, dim='id')
#%%
name = [i.name.split('_')[4] for i in hist_pr]
hist=hist.assign_coords(id=name)
#%%
ssp245=ssp245.assign_coords(id=name)
ssp585=ssp585.assign_coords(id=name)
#%%
hist.to_netcdf(r'H:\CMIP6 - SEA\mme\pr_hist_monthly_1998_2014.nc')
ssp245.to_netcdf(r'H:\CMIP6 - SEA\mme\pr_ssp245_monthly_2015_2100.nc')
ssp585.to_netcdf(r'H:\CMIP6 - SEA\mme\pr_ssp585_monthly_2015_2100.nc')
#%%
hist_y = hist.resample(time='AS').sum(skipna=False)
ssp245_y = ssp245.resample(time='AS').sum(skipna=False)
ssp585_y = ssp585.resample(time='AS').sum(skipna=False)

hist_mean = hist_y.mean(dim=['lat', 'lon'])
ssp245_mean = ssp245_y.mean(dim=['lat', 'lon'])
ssp585_mean = ssp585_y.mean(dim=['lat', 'lon'])

hist_mean_id = hist_mean.mean(dim='id')
ssp245_mean_id = ssp245_mean.mean(dim='id')
ssp585_mean_id = ssp585_mean.mean(dim='id')
#%%
diff245 = ssp245_mean - hist_mean_id.mean()
diff585 = ssp585_mean - hist_mean_id.mean()
#%%
ax = plt.axes()
# hist_mean_id.plot(ax=ax, color='blue')
diff245.mean(dim='id').plot(ax=ax, color='purple')
diff585.mean(dim='id').plot(ax=ax, color='red')
plt.xlabel('Year')
# plt.ylabel('ΔTmean (°C)')
# plt.ylabel('ΔPmean (%)')

# ax.fill_between(hist_mean.time, hist_mean.max(dim='id'), hist_mean.min(dim='id'), facecolor='lightblue', alpha=0.5)
ax.fill_between(diff245.time, diff245.max(dim='id'), diff245.min(dim='id'), facecolor='violet', alpha=0.5)
ax.fill_between(diff585.time, diff585.max(dim='id'), diff585.min(dim='id'), facecolor='red', alpha=0.3)
plt.show()
#%%
def diff_tmean(model, obs):
    return model - obs


def diff_pr(model, obs):
    return (model - obs) / obs * 100


ssp245_y = ssp245.resample(time='AS').sum(skipna=False)
ssp585_y = ssp585.resample(time='AS').sum(skipna=False)

ssp245_mean = ssp245_y.mean(dim=['lat', 'lon'])
ssp585_mean = ssp585_y.mean(dim=['lat', 'lon'])
# %%
diff_map_245 = diff_pr(ssp245_y, cru_mean)
diff_map_585 = diff_pr(ssp585_y, cru_mean)

diff_map_245_mme = diff_map_245.mean(dim='id')
diff_map_585_mme = diff_map_585.mean(dim='id')

diff_245 = diff_map_245.mean(dim=['lat', 'lon'])
diff_585 = diff_map_585.mean(dim=['lat', 'lon'])

# %%
near_245 = ut.select_year(diff_map_245_mme, 2015, 2039)
mid_245 = ut.select_year(diff_map_245_mme, 2040, 2069)
far_245 = ut.select_year(diff_map_245_mme, 2070, 2099)

near_585 = ut.select_year(diff_map_585_mme, 2015, 2039)
mid_585 = ut.select_year(diff_map_585_mme, 2040, 2069)
far_585 = ut.select_year(diff_map_585_mme, 2070, 2099)
# %%
da_ls = [near_245, mid_245, far_245, near_585, mid_585, far_585]
names = [
    'SSP2-4.5 (Near-future)',
    'SSP2-4.5 (Mid-future)',
    'SSP2-4.5 (Far-future)',
    'SSP5-8.5 (Near-future)',
    'SSP5-8.5 (Mid-future)',
    'SSP5-8.5 (Far-future)'
]
#%%
new_ds = xr.concat([da.mean(dim='time').assign_coords(name=f'({chr(ord("a") + i)}) {names[i]}') for i, da in enumerate(da_ls)], dim='name')
# %%
# diff_245 = diff_pr(ssp245_mean, cru_mean)
# diff_585 = diff_pr(ssp585_mean, cru_mean)

diff_245_mean = diff_245.mean(dim='id')
diff_585_mean = diff_585.mean(dim='id')

# %%
ax = plt.axes()
diff_245_mean.plot(ax=ax, color='purple')
diff_585_mean.plot(ax=ax, color='red')
plt.xlabel('Year')
# plt.ylabel('ΔTmean (°C)')
plt.ylabel('ΔPmean (%)')

ax.fill_between(diff_245_mean.time, diff_245.max(dim='id'), diff_245.min(dim='id'), facecolor='violet', alpha=0.5)
ax.fill_between(diff_585_mean.time, diff_585.max(dim='id'), diff_585.min(dim='id'), facecolor='red', alpha=0.3)
plt.show()
# %%
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12

#%%
import matplotlib.pyplot as plt
import util.preprocess as pre
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
#%%
def mf_plot(ds, levels=None):
    size = 23
    p = ds.plot(col='name', col_wrap=3,
                # cmap='jet',
                transform=ccrs.PlateCarree(),
                aspect=1,
                figsize=[2*x for x in plt.rcParams["figure.figsize"]],
                # levels=levels,
                cbar_kwargs={
                    'spacing': 'proportional',
                    'orientation': 'horizontal',
                    'shrink': 0.6,
                    'label': '',
                    'aspect': 40,
                    'anchor': (0.5, 1.7),
                },
                subplot_kws={
                    'projection': ccrs.PlateCarree()
                },
                )
    for i, ax in enumerate(p.axes.flat):
        ax.set_title(ds['name'].values[i])
        ax.coastlines()
        ax.add_feature(country_borders, edgecolor='darkgray')
        ax.set_extent([92.5, 142.5, -12.5, 24.5], crs=ccrs.PlateCarree())

    plt.show()
    plt.clf()
    plt.close()
mf_plot(new_ds)