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
    # n_ds = n_ds.interp(g4_lat_1=new_lat, g4_lon_2=new_lon)

    return n_ds


# %%

open_option = {
    # 'chunks': {'time': 2000},
    'concat_dim': 'initial_time0_hours',
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
# %%
mf = xr.open_mfdataset(paths)
# %%
mask_lon = ut.select_range(mf['g4_lon_2'], lon_bnds[0], lon_bnds[1])
mask_lat = ut.select_range(mf['g4_lat_1'], lat_bnds[0], lat_bnds[1])
# %%
n_mf = mf.isel(g4_lat_1=mask_lat, g4_lon_2=mask_lon, drop=True)
# %%
from nco import Nco

nco = Nco()
nco.ncap2()
out_path / paths[0].name + '.nc'

# %%
i = 1
s = len(paths)
err = []
for path in paths:
    out = out_path / path.name.replace('.man', '.nc')
    print(f'File[{i}/{s}]: {out}')
    i += 1
    # os.makedirs(out_path, exist_ok=True)
    try:
        with xr.open_dataset(path) as dset:
            dset = preprocess3(dset)
            dset.to_netcdf(out)
    except:
        if out.exists():
            os.remove(out)
        err.append((i, out))
        print('\terror')
