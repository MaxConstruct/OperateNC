# %%
import warnings

import xarray as xr
import numpy as np
from pathlib import Path
import util.netcdf_util as ut
import util.preprocess as pre
import matplotlib.pyplot as plt
from os import startfile as st

warnings.filterwarnings("ignore")

root = Path(r'H:\Observation\Raw Data\SA-OBS')
paths = sorted(list(root.iterdir()))
out = Path(r'H:\Observation\Raw Data (SEA)\[SEA] SA-OBS')
if not out.exists():
    out.mkdir()
    # %%

lat_attr = {
    'standard_name': 'latitude',
    'long_name': 'latitude',
    'units': 'degrees_north'
}

lon_attr = {
    'standard_name': 'longitude',
    'long_name': 'longitude',
    'units': 'degrees_east',
}
pr_attr = {
    'standard_name': 'precipitation_amount',
    'long_name': 'Daily precipitation',
    'units': 'mm',
}
#%%
def clean_file(ds):

    ds = ds.rename({'datetime': 'time', 'precip': 'pr'})
    ds.lat.attrs = lat_attr
    ds.lon.attrs = lon_attr
    ds.pr.attrs = pr_attr

    return ds.pr
    # ds = ds.rename({'precipitation': 'pr'
    #                 #    ,
    #                 # 'latitude': 'lat',
    #                 # 'longitude': 'lon'
    #                 }
    #                )
    # da = xr.decode_cf(ds).pr
    # re = pre.crop_sea(da)
    # return re.squeeze(drop=True).T.expand_dims(time=re.time)
    # return pre.regrid_sea(da)
    # return da.interp(lat=new_lat, lon=new_lon)


# %%
error = []
size = len(paths)
for i, p in enumerate(paths):
    try:
        with xr.open_dataset(p, decode_cf=False) as ds:
            re = clean_file(ds)
            print(i + 1, size, p, '\r', flush=True, end='')
            re.to_netcdf(out / ('SEA_' + p.name))
    except Exception as e:
        print('Error')
        error.append([i, p, str(e)])

print('Done.')
# %%
pds = sorted(list(out.iterdir()))
out2 = Path(r'H:\Observation\Cleaned Data\pr\daily 1998-2014')
# %%
mf_ds = xr.open_mfdataset(pds,
                          concat_dim='time',
                          # preprocess=clean_file,
                          )
mf_da = clean_file(mf_ds)
#%%
print('Loading...')
mf_da = mf_da.load()
print('Saving...')
mf_da.to_netcdf(out2 / 'persiann-cdr_pr_1998_2014.nc')
print('Done.')
# %%

ds = xr.open_dataset(r'H:\Observation\Raw Data (SEA)\[SEA] SA-OBS\tg_0.25deg_reg_v2.0_saobs.nc')
#%%
ds = ds.rename({'longitude': 'lon', 'latitude': 'lat', 'tg': 'tmean'})
da = ds.tmean

#%%
clean_dir = Path(r'H:\Observation\Cleaned Data')
daily = 'daily 1998-2014'
monthly = 'monthly 1998-2014'
mean = 'mean annual 1998-2014'
save_var = 'tmean'
ds_name = 'sa-obs'.lower()

ds_day_cut = pre.crop_sea(ut.select_year(da, 1998, 2014))
# %%
# ds_monthly = ds_day_cut.resample(time='M').sum(skipna=False)
# ds_mean = ds_monthly.resample(time='A').sum(skipna=False).mean(dim='time', skipna=True)

ds_monthly = ds_day_cut.resample(time='M').mean(skipna=True)
ds_mean = ds_monthly.mean(dim='time', skipna=True)

#%%
ds_day_cut.to_netcdf(clean_dir / save_var / daily / f'{ds_name}_{save_var}_daily_1998_2014.nc')
ds_monthly.to_netcdf(clean_dir / save_var / monthly / f'{ds_name}_{save_var}_monthly_1998_2014.nc')
ds_mean.to_netcdf(clean_dir / save_var / mean / f'{ds_name}_{save_var}_annual_1998_2014.nc')


