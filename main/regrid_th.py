import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature

import os
from main.util import plot, shift_to_180, sea_dataset, to_celsius

# %%
sample_path = r'H:\CMIP6 - SEA\IPSL-CM6A-LR\tasmax\historical\SEA_tasmax_day_IPSL-CM6A-LR_historical_r1i1p1f1_gr_18500101-20141231.nc'
sea = sea_dataset()
# %%
ds = xr.open_dataset(sample_path)
# %%
ds = ds.isel(time=slice(None, 2))
ds = to_celsius(shift_to_180(ds), attr='tasmax')
# %%
# ds = shift_to_180(ds)
min_lon = 97
max_lon = 106

min_lat = 5
max_lat = 21

resolution = 0.09

d_lon = [min_lon + i * resolution for i in range(np.int(np.ceil((max_lon - min_lon) / resolution)))]
d_lat = [min_lat + i * resolution for i in range(np.int(np.ceil((max_lat - min_lat) / resolution)))]
new_lon = xr.DataArray(d_lon, coords=[('lon', d_lon)], attrs=sea['lon'].attrs)
new_lat = xr.DataArray(d_lat, coords=[('lat', d_lat)], attrs=sea['lat'].attrs)
# %%
new_ds = ds.interp(lat=new_lat, lon=new_lon)
# %%
# plot(ds['tasmax'], savefig='sea_25km.pdf', country_border=True)
# plot(new_ds['tasmax'], savefig='th_9km.pdf', country_border=True)

#%%
plot(new_ds['tasmax'], center=20)
#%%
plot(ds['tasmax'], center=20, cmap='viridis')

#%%
