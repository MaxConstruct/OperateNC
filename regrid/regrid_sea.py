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
from util.file_util import wsl_paths, wsl_path

logging.basicConfig(filename=wsl_path(r'C:\Users\DEEP\PycharmProjects\NCFile\log\error.log'),
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )

warnings.filterwarnings("ignore")
# %%

output_root = CMIP6_SEA

variables = ['pr', 'tasmax', 'tasmin', 'mrro']
time_labels = ['historical', 'ssp245', 'ssp585']


# %%
def within_time_range(_name: str, limit=21001231):
    """
    Check if file is within acceptable time range.

    :param limit: time bound for checking. Default is 21001231 (31 Dec 2100)
    :param _name: CMIP6 name of the file which is, have a time range at the end of a name
            Example: pr_day_MRI-ESM2-0_ssp585_r1i1p1f1_gn_20150101-20641231.nc
    :return: True if given name is in the acceptable range otherwise, False.
    """
    return int(_name.split('_')[-1].split('-')[-1].replace('.nc', '')) <= limit


# %%



# %%
def file_name_from_merge(_paths):
    """
    Get new name from from given list of paths
    example:
        [foo_1908-1909.nc, foo_1910-1911.nc, ..., foo_2019-2020.nc]
        will return: 'foo_1908-2020.nc'
    :param _paths: list that contain names
    :return: New formatted name
    """
    p = str(_paths[0].name)
    return 'SEA_{}_{}.nc'.format('_'.join(p.split('_')[:-1]), ut.time_range(_paths))


# %%
def generate_merge_path(root_path, models, variables, time_label):
    """
    function that generate path for merging model in time-axis from given parameters
    for example:
        root_path = 'root', models = ['name1', 'name2'], variables = ['pr'], time_label = ['ssp245']
    It will generate:
        ['root/name1/pr/ssp245', 'root/name2/pr/ssp245']

    :param root_path: root path that contains all model
    :param models: name lists of all model in root_path path
    :param variables: variable name in each model | ex. ['pr', 'tasmax', 'tasmin', 'mrro']
    :param time_label: label id in model | ex. ['historical', 'ssp245', 'ssp585']
    :return: lists of path in format [root_path]/[model_name]/[variable_name]/[time_label]
    """
    ls = []
    for model in models:
        for var in variables:
            path_model_var = os.path.join(root_path, model, var)
            if os.path.exists(path_model_var):
                for lab in time_label:
                    path_model_var_lab = os.path.join(path_model_var, lab)
                    if os.path.exists(path_model_var_lab):
                        ls.append(path_model_var_lab)
    return np.array(ls)


# %%
sea_dataset = ut.sea_dataset()
new_lon = sea_dataset['lon']
new_lat = sea_dataset['lat']

# boundary for south east asia
lat_bnds = [-12, 24.5]
lon_bnds = [92.5, 142.5]


# %%
def preprocess(ds):
    """
    Preprocess for opening file option, in this case, regridding.
    :param ds: dataset
    :return: processed dataset
    """
    return ds.interp(lat=new_lat, lon=new_lon)


def preprocess2(data):
    """
    New algorithm for preprocessing. It will crop dataset before regridding
    make it faster for interpolation.

    This method is new and be waiting for testing performance.

    :param data: xarray.Dataset
    :return: processed dataset
    """
    n_ds = ut.crop_dataset_from_bound(data, lon_bound=lon_bnds, lat_bound=lat_bnds)
    return n_ds.interp(lat=new_lat, lon=new_lon)


# %%
# Get directory path, 3 level from root.
#
# root /  *  /   *    /    *
# CMIP6/model/variable/time_label
list_path = [i for i in CMIP6_PATH.glob('*/*/*')]

# Old code. It more convenient and easy to navigate using glob
# list_path = generate_merge_path(CMIP6_PATH,
#                                 CMIP6_MODEL_NAME,
#                                 variables,
#                                 time_labels)

# %%

skip = wsl_paths([
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\tasmax\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\tasmin\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\mrro\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\pr\ssp585'
])

# %%
open_option = {
    'chunks': {'time': 2000},
    'concat_dim': 'time',
    'engine': 'h5netcdf'
}
save_option = {
    'encoding': {
        var: {'compression': 'gzip', 'compression_opts': 5}
        for var in ['pr', 'lon_bnds', 'lat_bnds', 'time_bnds']
    },
    'engine': 'h5netcdf'
}

# %%
# currently working path
working_path = list_path
size = len(working_path)

count = 1
error = []
for src_path in working_path:

    # Get all file path in directory that satisfy time limit. In this case, 31 Dec 2100.
    # File that has time range more than limit will not include in the list.
    paths = [p for p in src_path.iterdir() if within_time_range(str(p), limit=21001231)]

    # Checking if path is not in skipping list
    if str(src_path) not in skip and len(paths) != 0:

        # Get new merged file name
        new_name = file_name_from_merge(paths)

        # New path for saving file
        out_path = Path(str(src_path).replace(str(CMIP6_PATH), str(output_root)), new_name)

        # It will skip this file if file already exist in output directory
        print(f'{bcolors.OKBLUE}File[{count}/{size}]:', out_path)
        if not out_path.exists():
            print(f'{bcolors.OKBLUE}File[{count}/{size}]:', out_path)
            status = ut.merge_regrid(paths=paths,
                                     out_dst=out_path,
                                     preprocess=preprocess,
                                     _open_option=open_option,
                                     _save_option=save_option
                                     )

            # If operation is not success logging an error information.
            if not status[0]:
                error_info = [count, new_name, src_path, out_path]
                logging.error('File[{count}/{size}]: ' + ' '.join(error_info))
                error.append(error_info)

                # Delete an error file
                if out_path.exists():
                    out_path.unlink(missing_ok=True)
        count += 1

print('All Done')
