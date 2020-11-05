# %%
import sys
sys.path.extend(['C:\\Users\\DEEP\\PycharmProjects\\NCFile', 'C:/Users/DEEP/PycharmProjects/NCFile'])

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

aph = xr.open_dataset(r'H:\Observation\Raw Data (SEA)\[SEA] Aphrodite 1901 & 1808\APHRO_MA_TAVE_025deg_V1808_1961-2015_Temp.nc')
sa = xr.open_dataset(r'H:\Observation\Raw Data (SEA)\[SEA] SA-OBS\tg_0.25deg_reg_v2.0_saobs.nc')

aph_da = aph.tave
sa_da = sa.tg
sa_da = sa_da.rename({'longitude': 'lon', 'latitude': 'lat'})
crop_sa_da = pre.crop_sea(sa_da)
crop_aph_da = pre.crop_sea(aph_da)

time1 = [ut.select_year(crop_sa_da, 1998, 2014), ut.select_year(crop_aph_da, 1998, 2014)]
time2 = [ut.select_year(crop_sa_da, 1981, 2007), ut.select_year(crop_aph_da, 1981, 2007)]
#%%
mean_time1 = [pre.crop_sea(i.mean(dim='time')) for i in time1]
mean_time2 = [pre.crop_sea(i.mean(dim='time')) for i in time2]
#%%
time = mean_time2
#%%
fig, axes = plt.subplots(ncols=3,
                             nrows=1,
                             figsize=(16, 5),
                             # subplot_kw={
                             #     'projection': ccrs.PlateCarree()
                             # },
                             tight_layout=False)
faxes = axes.flatten()
time[0].plot(ax=faxes[0], levels=np.arange(5, 31, 0.01), cmap='jet')
time[1].plot(ax=faxes[1], levels=np.arange(5, 31, 0.01), cmap='jet')
(time[0] - time[1]).plot(ax=faxes[2])

faxes[0].set_title('sa-obs')
faxes[1].set_title('aphrodite')
faxes[2].set_title('different')
# for ax in faxes:
#     ax.coastlines()

# plt.title('1998-2014')
plt.title('1981-2007')
plt.show()
plt.clf()
plt.close()
