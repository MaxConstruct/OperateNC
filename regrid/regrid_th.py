"""
** Working in progress **
Regrid and crop thailand only
"""

import xarray as xr
# import xesmf as xe
import numpy as np
import util.netcdf_util as ut
from util.file_util import wsl_path

# %%
min_lon = 97
max_lon = 106

min_lat = 5
max_lat = 21

resolution = 0.09

lon_bound = [min_lon, max_lon]
lat_bound = [min_lat, max_lat]
# %%
sample_path = r'/mnt/h/CMIP6 - SEA/IPSL-CM6A-LR/tasmax/historical/SEA_tasmax_day_IPSL-CM6A-LR_historical_r1i1p1f1_gr_18500101-20141231.nc'

# %%
ds = ut.sea_dataset()
# %%
# ds_out = xr.Dataset({'lat': (['lat'], np.arange(16, 75, 1.0)),
#                      'lon': (['lon'], np.arange(200, 330, 1.5)),
#                     }
#                    )
crop_ds = ut.crop_dataset_from_bound(ds, lon_bound=lon_bound, lat_bound=lat_bound)
# %%

