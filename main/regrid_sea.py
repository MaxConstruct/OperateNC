# %%
import cftime
import xarray as xr
import os
import numpy as np
import warnings
import logging
from os.path import exists, join
import traceback
from distributed.deploy.old_ssh import bcolors
from hurry.filesize import size as read_size

logging.basicConfig(filename=r'C:\Users\DEEP\PycharmProjects\NCFile\main\log\error.log',
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )

warnings.filterwarnings("ignore")
# %%
READY = False

ROOT_PATH = r'H:\CMIP6 - 27 Models'
OUTPUT_PATH = r'H:\CMIP6 - SEA'

SEA_MODEL = r'H:\CMIP6 - Test\SAMPLE - pr_SEA-25_HadGEM2-AO_historical_r1i1p1_WRF_v3-5_day_197101-198012.nc'

VARIABLES = ['pr', 'tasmax', 'tasmin', 'mrro']
TIME_LABELS = ['historical', 'ssp245', 'ssp585']

ALL_MODEL = np.array(os.listdir(ROOT_PATH))


# %%
def shift_to_180(ds):
    return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')


def to_celsius(ds, attr):
    ds[attr] = ds[attr] - 273.15
    return ds


def crop_to_model(sample, model):
    new_lon, new_lat = get_coord(sample)
    return model.sel(lat=new_lat, lon=new_lon, method='nearest')


def get_coord(ds):
    return ds['lon'], ds['lat']


def interp_crop_to_model(sample, model):
    new_lon, new_lat = get_coord(sample)
    return model.interp(lat=new_lat, lon=new_lon)


# %%
def time_check(file_name: str, limit=21001231):
    """
    Check if file is within acceptable time range.

    :param limit: time bound for checking. Default is 21001231 (31 Dec 2100)
    :param file_name: CMIP6 name of the file which is, have a time range at the end of a name
            Example: pr_day_MRI-ESM2-0_ssp585_r1i1p1f1_gn_20150101-20641231.nc
    :return: True if given name is in the acceptable range, Example. 21001231 (31 Dec 2100) otherwise, False.
    """
    return int(file_name.split('_')[-1].split('-')[-1].replace('.nc', '')) <= limit


# %%

def list_file_abs_path(path):
    return [os.path.join(path, i) for i in os.listdir(path) if time_check(i)]


def time_range(paths):
    t0 = paths[0].split('_')[-1].split('-')[0]
    t1 = paths[-1].split('_')[-1].split('-')[-1].replace('.nc', '')
    return '{}-{}'.format(t0, t1)


# %%
def file_name(paths):
    p = os.path.basename(paths[0])
    return 'SEA_{}_{}.nc'.format('_'.join(p.split('_')[:-1]), time_range(paths))


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
sea_dataset = xr.open_dataset(SEA_MODEL)
new_lon = sea_dataset['lon']
new_lat = sea_dataset['lat']


# %%
def preprocess(ds):
    return ds.interp(lat=new_lat, lon=new_lon)


# %%
def regrid(paths, out_dst,
           comp=None,
           h5netcdf_engine=True,
           ):
    if comp is None:
        comp = {'compression': 'gzip', 'compression_opts': 5}
    try:
        if h5netcdf_engine:
            with xr.open_mfdataset(paths,
                                   preprocess=preprocess,
                                   concat_dim='time',
                                   parallel=True,
                                   chunks={'time': 3000},
                                   engine='h5netcdf',
                                   # decode_cf=True
                                   ) as mf_dataset:
                encoding = {var: comp for var in mf_dataset.data_vars}
                os.makedirs(os.path.dirname(out_dst), exist_ok=True)
                mf_dataset.to_netcdf(out_dst,
                                     engine='h5netcdf',
                                     encoding=encoding
                                     )
        else:
            with xr.open_mfdataset(paths,
                                   preprocess=preprocess,
                                   concat_dim='time',
                                   parallel=True,
                                   chunks={'time': 3000},
                                   # decode_cf=True
                                   ) as mf_dataset:
                encoding = {var: comp for var in mf_dataset.data_vars}
                os.makedirs(os.path.dirname(out_dst), exist_ok=True)
                mf_dataset.to_netcdf(out_dst,
                                     encoding=encoding
                                     )

    except Exception as ex:
        print('\t\t', f"{bcolors.FAIL}Error {bcolors.ENDC} {str(ex)}")


# %%

list_path = generate_merge_path(ROOT_PATH, ALL_MODEL, VARIABLES, TIME_LABELS)
# %%
size = len(list_path)

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
path = skip
# %%
for p in path:
    paths = list_file_abs_path(p)
    name = file_name(paths)
    out = join(p.replace(ROOT_PATH, OUTPUT_PATH), name)
    print(p, name)
    regrid(paths=paths, out_dst=out, comp={'zlib': True, 'complevel': 5}, h5netcdf_engine=False)

                                                                                                                    # %%
# count = 1
# error = []
# error_src_ls = []
# # ls_size = []
# # ls_name = []
# # sorted(ls_size, reverse=True)
# for src in focus_file:
#     if src[1] not in skip:
#         paths = list_file_abs_path(src[1])
#         name = file_name(paths)
#         out_dir_path = src[1].replace(ROOT_PATH, OUTPUT_PATH)
#         out_file_path = join(out_dir_path, name)
#
#         if not exists(out_file_path):
#             count += 1
#
#             print(f'{bcolors.OKBLUE}File[{src[0]}/{size}]:', out_file_path)
#             # with xr.set_options(enable_cftimeindex=True):
#             error = regrid(paths=paths, out_dst=out_file_path, name=name)
#
# print('All Done')
# print(count)