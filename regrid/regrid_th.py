
"""
Regrid and crop thailand only
"""

import xarray as xr
import numpy as np
import util.netcdf_util as ut

#%%
min_lon = 97
max_lon = 106

min_lat = 5
max_lat = 21

resolution = 0.09

lon_bound = [min_lon, max_lon]
lat_bound = [min_lat, max_lat]
# %%
sample_path = r'H:\CMIP6 - SEA\IPSL-CM6A-LR\tasmax\historical\SEA_tasmax_day_IPSL-CM6A-LR_historical_r1i1p1f1_gr_18500101-20141231.nc'

#%%
sea = ut.sea_dataset()


#%%
new_lon, new_lat = ut.new_coord_array(lat_bound=lat_bound, lon_bound=lon_bound, res=resolution)
#%%
ds = ut.crop_dataset_from_bound(sea, lon_bound=lon_bound, lat_bound=lat_bound)
#%%
new_ds = ds.interp(lat=np.arange(5, 21, 0.09), lon=np.arange(97, 106, 0.09))

#%%
new_ds.to_netcdf(r'H:\CMIP6 - Test\test_re4.nc', encoding={var: {'zlib': True, 'complevel': 5} for var in new_ds.data_vars})
# %%
# plot(ds['tasmax'], savefig='sea_25km.pdf', country_border=True)
# plot(new_ds['tasmax'], savefig='th_9km.pdf', country_border=True)
#
# plot(new_ds['tasmax'], center=20)
# plot(ds['tasmax'], center=20, cmap='viridis')