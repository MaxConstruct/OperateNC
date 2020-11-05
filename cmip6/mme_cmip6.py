# %%
from pathlib import Path
from setting import CMIP6_SEA
import xarray as xr
import numpy as np
import util.netcdf_util as ut
from os import startfile as st

# %%
skip = """
CESM2
CESM2-WACCM
CNRM-CM6-1-HR
EC-Earth3-Veg
FGOALS-f3-L
MPI-ESM1-2-HR
UKESM1-0-LL
""".split()

#%%
def get_paths(variable='', experiment='', _skip=None):
    if _skip is None:
        _skip = []

    _paths = []
    CMIP6_SEA_IN_USED = Path(r'H:\CMIP6 - SEA\In Use')
    for p in CMIP6_SEA_IN_USED.glob('*/*/*/*'):
        if p.parent.parent.parent.name not in _skip and \
                p.parent.parent.name == variable and \
                p.parent.name == experiment:
            _paths.append(p)
    _paths.sort()
    return _paths


# %%
# %%

def select_datetime(ds):
    return ds.sel(time=(
            ~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))
            & ds.time.dt.year.isin(np.arange(1998, 2015))), drop=True
    )


def shapes(paths):
    for i in paths:
        ds = xr.open_dataarray(i)
        print(ds.shape)
        ds.close()

#%%
def units(paths):
    for i in paths:
        ds = xr.open_dataarray(i)
        print(ds.attrs['units'])
        ds.close()
# %%
ssp245 = Path(r'H:\CMIP6 - SEA\Cleaned\ssp245')
ssp585 = Path(r'H:\CMIP6 - SEA\Cleaned\ssp585')


def save_to(path: Path, name):
    if not path.exists():
        path.mkdir()
    return path / name


# %%
for var in ['pr', 'tasmax', 'tasmin']:
    ex_time = 'ssp585'
    raw_paths = get_paths(variable=var, experiment=ex_time, _skip=skip)
    size = len(raw_paths)
    # print(*raw_paths, sep='\n')
    print('\n', var, ex_time, '\n----------')
    for i, path in enumerate(raw_paths):
        print(i + 1, size, path.name)
        new_ds = xr.load_dataset(path)[var]
        # new_ds = ut.select_year(ds.tasmin, 1998, 2014)
        new_ds_noleap = new_ds.sel(time=~((new_ds.time.dt.month == 2) & (new_ds.time.dt.day == 29)))
        new_ds_noleap.to_netcdf(save_to(ssp585 / f'decode_cmip_{var}_{ex_time}_2015_2100_noleap', 'DECODE_' + path.name))
        new_ds.close()

# %%
root = Path(r'H:\CMIP6 - SEA\Cleaned\historical')
# pr_paths = sorted([i for i in Path(ut.test_path('decode_cmip_pr_1998_2015_noleap')).iterdir()])
tasmax_paths = sorted(list((root / 'decode_cmip_tasmax_1998_2014_noleap').iterdir()))
tasmin_paths = sorted(list((root / 'decode_cmip_tasmin_1998_2014_noleap').iterdir()))

# %%
pr_attr = {
    'standard_name': 'precipitation_flux',
    'long_name': 'Precipitation',
    'units': 'kg m-2 s-1'
}
t_attr = {
    'standard_name': 'air_temperature',
    'long_name': 'Maximum Near-Surface Air Temperature',
    'units': 'Celsius'
}
tmean_attr = {
    'standard_name': 'air_temperature',
    'long_name': 'Mean Near-Surface Air Temperature',
    'units': 'Celsius'
}

# %%
pr_basic_qm_hist_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_basic_qm\historical').iterdir()))
pr_mod_qm_hist_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_mod_qm\historical').iterdir()))

# tmean_bias_245_path = sorted(list(Path(r'H:\CMIP6 - Biased\tmean\ssp245').iterdir()))
pr_basic_qm_245_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_basic_qm\ssp245').iterdir()))
pr_mod_qm_245_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_mod_qm\ssp245').iterdir()))

# tmean_bias_585_path = sorted(list(Path(r'H:\CMIP6 - Biased\tmean\ssp585').iterdir()))
pr_basic_qm_585_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_basic_qm\ssp585').iterdir()))
pr_mod_qm_585_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_mod_qm\ssp585').iterdir()))
#%%

def mme_dataset(paths, to_celsius=False):
    base = xr.open_dataarray(paths[0])
    base_coords = base.coords
    n = len(paths)
    s = np.zeros_like(base)
    for i, p in enumerate(paths):
        print(i + 1, n, p.name)
        da = xr.load_dataarray(p)
        if to_celsius:
            s += da.values - 273.15
        else:
            s += da.values
        da.close()
    s /= n
    base.close()
    mme = xr.DataArray(
        avg_arr,
        {
            'time': base_coords['time'].values,
            'lat': base_coords['lat'].values,
            'lon': base_coords['lon'].values
        },
        dims=['time', 'lat', 'lon'],
        # attrs=t_attr,
        name='pr'
    )
    return mme


# %%
print('Averaging...')
avg_arr, coords = mme_dataset(tasmax_paths, to_celsius=True)

print('Creating new dataset...')
name = 'pr'
mme = xr.DataArray(
    avg_arr,
    {
        'time': coords['time'].values,
        'lat': coords['lat'].values,
        'lon': coords['lon'].values
    },
    dims=['time', 'lat', 'lon'],
    # attrs=t_attr,
    name=name
)
print('Saving MME daily...')
mme.to_netcdf(ut.test_path(f'SEA_MME_{name.upper()}_DAILY.nc'))
print('Saving MME monthly...')
mme_monthly = mme.resample(time='m').mean(keep_attrs=True)
mme_monthly.to_netcdf(ut.test_path(f'SEA_MME_{name.upper()}_MONTHLY.nc'))
print('Done.')

#%%
