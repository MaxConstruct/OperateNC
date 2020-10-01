# %%
from time import sleep

import xarray as xr
import numpy as np
from pathlib import Path
import util.netcdf_util as ut
import util.preprocess as pre
import matplotlib.pyplot as plt
from os import startfile as st

cmorph_path = Path(r'H:\Observation\Raw Data\CMORPH')

sea_cmorph_path = Path(r'H:\Observation\Raw Data (SEA)\[SEA] CMORPH')
paths = sorted(list(cmorph_path.iterdir()))

# %%
def clean_file(ds):
    da = ut.shift_to_180(ds.drop_dims('nv').rename({'cmorph': 'pr'})).pr
    return pre.crop_sea(da)
    # return da.interp(lat=new_lat, lon=new_lon)

#%%

error = []
size = len(paths)
for i, p in enumerate(paths):
    try:
        with xr.open_dataset(p) as ds:
            re = clean_file(ds)
            print(i+1, size, p, '\r', flush=True, end='')
            re.to_netcdf(sea_cmorph_path / ('SEA_' + p.name))
    except Exception as e:
        print('Error')
        error.append([i, p, str(e)])

print('Done.')
#%%
sea_paths = sorted(list(sea_cmorph_path.iterdir()))
mf_ds = xr.open_mfdataset(sea_paths,
                          concat_dim='time',
                          parallel=True
                          )

#%%
print('Saving.')
mf_ds.to_netcdf(sea_cmorph_path / 'SEA_CMORPH_Z_1998_2014.nc')
print('Done.')

#%%
clean_dir = Path(r'H:\Observation\Cleaned Data')
daily = 'daily 1998-2014'
monthly = 'monthly 1998-2014'
mean = 'mean annual 1998-2014'
save_var = 'pr'
ds_name = 'cmorph'

ds_day_cut = xr.open_dataset(r'H:\Observation\Cleaned Data\pr\daily 1998-2014\cmorph_pr_1998_2014.nc')
ds_monthly = ds_day_cut.resample(time='M').sum()
ds_mean = ds_monthly.resample(time='A').sum().mean(dim='time')

# ds_day_cut.to_netcdf(clean_dir / save_var / daily / f'{ds_name}_{save_var}_daily_1998_2014.nc')
ds_monthly.to_netcdf(clean_dir / save_var / monthly / f'{ds_name}_{save_var}_monthly_1998_2014.nc')
ds_mean.to_netcdf(clean_dir / save_var / mean / f'{ds_name}_{save_var}_annual_1998_2014.nc')