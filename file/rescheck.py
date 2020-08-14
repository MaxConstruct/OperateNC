import os
import numpy as np
import xarray as xr

root = 'I:\CMIP6'
variables = ['pr', 'tasmax', 'tasmin']
labels = ['hist-1950', 'ssp245', 'ssp585']


# for path, subdirs, files in os.walk(root):
#     if subdirs:
#         print(path, subdirs)
# %%
def diff(dset):
    return abs(dset[0].values - dset[1].values)


# %%
for var in variables:
    print(var, '-------------------------------')
    for lab in labels:
        models_path = os.path.join(root, var, lab)
        models = os.listdir(models_path)
        print('\t|', lab)
        for model_name in models:
            root_file_path = os.path.join(models_path, model_name)
            file_name = os.listdir(root_file_path)[0]
            file_path = os.path.join(root_file_path, file_name)
            # print("\t\t|", model_name, os.path.join(root_file_path, '*.nc'))
            # with nc.Dataset(file_path) as dset:
            #     # print(dset)
            with xr.open_dataset(file_path, decode_cf=False) as dset:
                print("\t\t{:5}\t{:5} --\t{:4}\t{:4}\t{}".format(dset.dims['lat'], dset.dims['lon'], diff(dset['lat']),
                                                                 diff(dset['lon']), model_name))

            #     lat = (dset.variables['lat'][:])
            #     delta(lat)
            #     lon = (dset.variables['lon'][:])
            #     delta(lon)
            #     # print(dset)
# %%
import os

def walk_through_files(path, file_extension='.nc'):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dirpath, filename)


root = 'I:\CORDEX-SEA\SEA-25'
# %%
for p in walk_through_files(root):
    with xr.open_dataset(p, decode_cf=False) as dset:
        print("\t\t{:5}\t{:5} --\t{:4}\t{:4}".format(dset.dims['lat'], dset.dims['lon'], diff(dset['lat']),
                                                     diff(dset['lon'])), p)
