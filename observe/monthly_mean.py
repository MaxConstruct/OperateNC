
import xarray as xr
import numpy as np
import pandas as pd
from observe import observe_path as op
import util.netcdf_util as ut
from pathlib import Path
#%%
clean_dir = Path(r'H:\Observation\Cleaned Data')
#%%
# # mf_path = sorted([i for i in Path(r'H:\Observation\Cleaned Data\monthly 1998-2014').iterdir()])
# #%%
# cru_p = Path(r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.pre.dat.nc')
# cru = xr.open_dataset(cru_p, drop_variables=['stn']).rename({'pre': 'pr'})
# #%%
# cru_pr = ut.select_year(cru.pr.copy(), 1998, 2014)
# #%%
# for i in range(len(cru_pr)):
#     cru_pr[i] /= cru.time.dt.days_in_month[i]
# #%%
#
# cru_pr.attrs = {
#     'long_name': 'precipitation',
#     'units': 'mm/day'
# }
# #%%
# cru_pr.to_netcdf(clean_dir / 'monthly 1998-2014' / 'cru_pr_monthly_1998_2014.nc')

#%%

