#%%
import xarray as xr
import matplotlib.pyplot as plt
import util.netcdf_util as ut
import numpy as np
from pathlib import Path

days = Path(r'H:\Observation\Cleaned Data\pr\daily 1998-2014\cpc_pr_1998_2014.nc')

ds = xr.open_dataarray(days)

