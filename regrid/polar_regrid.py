
"""
Trying to re-projection from rotate degree to lat-lon coordinate system.
This still in experiment since a given dataset (CORDEX-EAS) has no detail on
projecting information such as proj4 parameter or projection name.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
#%%
# Default value
# pole_longitude=0.0
# pole_latitude=90.0
# central_rotated_longitude=0.0
# globe=None
# pole_latitude = 77.61
# pole_longitude = -64.78
#
#
# proj4_params = [('proj', 'ob_tran'), ('o_proj', 'latlon'),
#                         ('o_lon_p', central_rotated_longitude),
#                         ('o_lat_p', pole_latitude),
#                         ('lon_0', 180 + pole_longitude),
#                         ('to_meter', math.radians(1))]

# #%%
# SEA = r'I:\Cordex-EAS-SEA\SEA\RegCM4-3\EC-EARTH\pr\hist\pr_SEA-22_ICHEC-EC-EARTH_historical_r1i1p1_ICTP-RegCM4-3_v4_day_1970010112-1970013112.nc'
# #%%
# x = xr.open_dataset(SEA, drop_variables='time_bnds')
