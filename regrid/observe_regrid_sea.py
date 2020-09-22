# %%
from pathlib import Path

import xarray as xr
import os
import numpy as np
import warnings
import logging

from distributed.deploy.old_ssh import bcolors

import util.netcdf_util as ut
from setting import CMIP6_PATH, CMIP6_SEA

logging.basicConfig(filename=r'C:\Users\DEEP\PycharmProjects\NCFile\log\error.log',
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )

warnings.filterwarnings("ignore")

# %%
sea_dataset = ut.sea_dataset()
new_lon = sea_dataset['lon']
new_lat = sea_dataset['lat']
# %%
# boundary for south east asia
lat_bnds = [-12, 24.5]
lon_bnds = [92.5, 142.5]
# %%
n_lon, n_lat = ut.new_coord_array(lon_bound=lon_bnds, lat_bound=lat_bnds, res=0.25, x_name='g4_lat_1',
                                  y_name='g4_lon_2')


# %%
def preprocess(data):
    return ut.crop_dataset_from_bound(data, lon_bound=lon_bnds, lat_bound=lat_bnds, x_name='lat', y_name='lon')


# %%
def preprocess2(data):
    n_ds = ut.crop_dataset_from_bound(data, lon_bound=lon_bnds, lat_bound=lat_bnds, x_name='g4_lat_1',
                                      y_name='g4_lon_2')
    n_ds = ut.kelvin_to_celsius(n_ds, 'TMAX_GDS4_HTGL')
    return n_ds.interp(g4_lat_1=new_lat, g4_lon_2=new_lon)


# %%
def preprocess3(data):
    mask_lon = ut.select_range(data['lon'], lon_bnds[0], lon_bnds[1])
    mask_lat = ut.select_range(data['lat'], lat_bnds[0], lat_bnds[1])

    n_ds = data.isel(lat=mask_lat, lon=mask_lon, drop=True)

    # n_ds = ut.kelvin_to_celsius(n_ds, 'TMAX_GDS4_HTGL')
    n_ds = n_ds.interp(lat=new_lat, lon=new_lon)

    return n_ds


# %%

open_option = {
    # 'chunks': {'time': 2000},
    'concat_dim': 'time',
    # 'engine': 'h5netcdf'
}
save_option = {
    # 'encoding': {
    #     var: {'compression': 'gzip', 'compression_opts': 5}
    #     for var in ['pr', 'lon_bnds', 'lat_bnds', 'time_bnds']
    # },
    # 'engine': 'h5netcdf'
}

# %%
# currently working path
working_path = Path(r'H:\Observation\TRMM_3B42_Daily (TMPA)\Dataset')

paths = [p for p in working_path.iterdir()]

out_path = Path(r'H:\Observation\[SEA] TRMM_3B42_Daily (TMPA)')
# %%
status = ut.merge_regrid(paths=paths,
                         out_dst=out_path,
                         preprocess=preprocess3,
                         _open_option=open_option
                         )

if not status[0]:
    print(status[1])

    if out_path.exists():
        os.remove(str(out_path))

print('All Done')

#%%

for i, p in enumerate(paths):
    out = out_path / p.name
    print(i+1, len(paths), p)
    ds = xr.load_dataset(p)
    try:
        n_ds = preprocess3(ds)
        n_ds.to_netcdf(out)
    except Exception as e:
        print('Error', str(e))
        if out.exists():
            os.remove(out)
    ds.close()

#%%

s = len(paths)
err = []
drop_var = [
 'randomError_cnt',
 'HQprecipitation_cnt',
 'randomError',
 'HQprecipitation',
 'IRprecipitation',
 'precipitation_cnt',
 'IRprecipitation_cnt'
]
for i, path in enumerate(paths):
    out = out_path / path.name.replace('.man', '.nc')
    print(f'File[{i}/{s}]: {out}')
    # os.makedirs(out_path, exist_ok=True)
    try:
        with xr.open_dataset(path, drop_variables=drop_var) as dset:
            dset = preprocess3(dset)
            dset.to_netcdf(out)
    except:
        if out.exists():
            os.remove(out)
        err.append((i, out))
        print('\terror')
