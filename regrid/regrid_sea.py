# %%
import xarray as xr
import os
import numpy as np
import warnings
import logging
from os.path import exists, join
from distributed.deploy.old_ssh import bcolors
import util.netcdf_util as ut

logging.basicConfig(filename=r'C:\Users\DEEP\PycharmProjects\NCFile\log\error.log',
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )

warnings.filterwarnings("ignore")
# %%

ROOT_PATH = r'H:\CMIP6 - 27 Models'
OUTPUT_PATH = r'H:\CMIP6 - SEA'

VARIABLES = ['pr', 'tasmax', 'tasmin', 'mrro']
TIME_LABELS = ['historical', 'ssp245', 'ssp585']

ALL_MODEL = np.array(os.listdir(ROOT_PATH))


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
def time_range(_paths):
    """
    Get time range from given list of paths
    example:
        [foo_1908-1909.nc, foo_1910-1911.nc, ..., foo_2019-2020.nc]
        will return: '1908-2020'

    :param _paths: list that contain names
    :return: New name with lowest time range to max
    """
    t0 = _paths[0].split('_')[-1].split('-')[0]
    t1 = _paths[-1].split('_')[-1].split('-')[-1].replace('.nc', '')
    return '{}-{}'.format(t0, t1)


# %%
def file_name(_paths):
    """
    Get new name from from given list of paths
    example:
        [foo_1908-1909.nc, foo_1910-1911.nc, ..., foo_2019-2020.nc]
        will return: 'foo_1908-2020.nc'
    :param _paths: list that contain names
    :return: New formatted name
    """
    p = os.path.basename(_paths[0])
    return 'SEA_{}_{}.nc'.format('_'.join(p.split('_')[:-1]), time_range(_paths))


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
            path_model_var = join(root_path, model, var)
            if exists(path_model_var):
                for lab in time_label:
                    path_model_var_lab = join(path_model_var, lab)
                    if exists(path_model_var_lab):
                        ls.append(path_model_var_lab)
    return np.array(ls)


# %%
sea_dataset = ut.sea_dataset()
new_lon = sea_dataset['lon']
new_lat = sea_dataset['lat']

# boundary for south east asia
lon_bnds = [-12, 24.5]
lat_bnds = [92.5, 142.5]


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
def merge_regrid(paths, out_dst, _open_option=None, _save_option=None):
    """
    Open multiple dataset from list of paths, then, merging and regridding
    and save processing file as netcdf.

    :param paths: list of paths of dataset to open and merge
    :param out_dst: path to save processed dataset
    :param _open_option: dictionary option pass to xarray.open_mfdatset
    :param _save_option: dictionary option pass to xarray.Dataset.to_netcdf
    :return: Status of the operation as tuple (boolean, message)
    If operation is success return (True, 'success') otherwise, (False, error message)
    """

    if _save_option is None:
        _save_option = {}
    if _open_option is None:
        _open_option = {}

    try:
        with xr.open_mfdataset(paths=paths,
                               preprocess=preprocess,
                               parallel=True,
                               **_open_option) as mf_dataset:
            os.makedirs(os.path.dirname(out_dst), exist_ok=True)
            mf_dataset.to_netcdf(out_dst, **_save_option)

    # Old code

        # if h5netcdf_engine:
        #     with xr.open_mfdataset(paths,
        #                            preprocess=preprocess,
        #                            concat_dim='time',
        #                            parallel=True,
        #                            chunks={'time': 3000},
        #                            engine='h5netcdf',
        #                            # decode_cf=True
        #                            ) as mf_dataset:
        #         encoding = {var: comp for var in mf_dataset.data_vars}
        #         os.makedirs(os.path.dirname(out_dst), exist_ok=True)
        #         mf_dataset.to_netcdf(out_dst,
        #                              engine='h5netcdf',
        #                              encoding=encoding
        #                              )
        # else:
        #     with xr.open_mfdataset(paths,
        #                            preprocess=preprocess,
        #                            concat_dim='time',
        #                            parallel=True,
        #                            chunks={'time': 3000},
        #                            # decode_cf=True
        #                            ) as mf_dataset:
        #         encoding = {var: comp for var in mf_dataset.data_vars}
        #         os.makedirs(os.path.dirname(out_dst), exist_ok=True)
        #         mf_dataset.to_netcdf(out_dst,
        #                              encoding=encoding
        #                              )

    except Exception as ex:
        print('\t\t', f"{bcolors.FAIL}Error {bcolors.ENDC} {str(ex)}")
        return False, str(ex)

    return True, 'success'


# %%

list_path = generate_merge_path(ROOT_PATH, ALL_MODEL, VARIABLES, TIME_LABELS)

# %%
skip = [
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\tasmax\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\tasmin\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\mrro\ssp585',
    r'H:\CMIP6 - 27 Models\CESM2-WACCM\pr\ssp585'
]

list_error_file = {
    'BCC-CSM2-MR':
        [r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\pr\historical',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\pr\ssp245',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\pr\ssp585',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmax\historical',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmax\ssp245',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmax\ssp585',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmin\historical',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmin\ssp245',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\tasmin\ssp585',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\mrro\historical',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\mrro\ssp245',
         r'H:\CMIP6 - 27 Models\BCC-CSM2-MR\mrro\ssp585'
         ],
    'CNRM':
        [r'H:\CMIP6 - 27 Models\CNRM-CM6-1\pr\historical',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1\pr\ssp245',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1\pr\ssp585',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1\mrro\historical',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1-HR\pr\historical',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1-HR\pr\ssp585',
         r'H:\CMIP6 - 27 Models\CNRM-CM6-1-HR\mrro\historical',
         r'H:\CMIP6 - 27 Models\CNRM-ESM2-1\pr\historical',
         r'H:\CMIP6 - 27 Models\CNRM-ESM2-1\pr\ssp245',
         r'H:\CMIP6 - 27 Models\CNRM-ESM2-1\pr\ssp585',
         r'H:\CMIP6 - 27 Models\CNRM-ESM2-1\mrro\historical'],
    'FGOALS':
        [r'H:\CMIP6 - 27 Models\FGOALS-f3-L\pr\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-f3-L\tasmax\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-f3-L\tasmin\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\pr\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\pr\ssp245',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\pr\ssp585',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmax\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmax\ssp245',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmax\ssp585',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmin\historical',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmin\ssp245',
         r'H:\CMIP6 - 27 Models\FGOALS-g3\tasmin\ssp585'
         ],
    'IPSL-CM6A-LR':
        [r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\pr\historical',
         r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\pr\ssp245',
         r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\pr\ssp585',
         r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\mrro\historical',
         r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\mrro\ssp245',
         r'H:\CMIP6 - 27 Models\IPSL-CM6A-LR\mrro\ssp585'
         ]
}
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

    # Checking if path is not in skipping list
    if src_path not in skip:

        # Get all file path in directory that satisfy time limit. In this case, 31 Dec 2100.
        # File that has time range more than limit will not include in the list.
        paths = ut.listdir_abs(src_path, condition=within_time_range(src_path, limit=21001231))

        # Get new merged file name
        name = file_name(paths)

        # New path for saving file
        out_path = join(src_path.replace(ROOT_PATH, OUTPUT_PATH), name)

        # It will skip this file if file already exist in output directory
        if not exists(out_path):
            print(f'{bcolors.OKBLUE}File[{count}/{size}]:', out_path)
            status = merge_regrid(paths=paths,
                                  out_dst=out_path,
                                  _open_option=open_option,
                                  _save_option=save_option
                                  )

            # If operation is not success logging an error information.
            if not status[0]:
                error_info = [count, name, src_path, out_path]
                logging.error('File[{count}/{size}]: '+' '.join(error_info))
                error.append(error_info)

                # Delete an error file
                if exists(out_path):
                    os.remove(out_path)

print('All Done')
