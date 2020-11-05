# %%
import numba
from pathlib import Path
import bias_correction as bias
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from bias_correction import BiasCorrection
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import util.netcdf_util as ut
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from bias_correction import XBiasCorrection
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# %%
def basic_quantile(obs_data, mod_data, sce_data):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])

    return sce_data + cor


def modified_quantile(obs_data, mod_data, sce_data):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])

    mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
    g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
    iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
    iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

    f = np.true_divide(iqr_obs_data, iqr_mod_data)
    cor = g * mid + f * (cor - mid)
    return sce_data + cor


def convert_neg_to_zero(da: xr.DataArray) -> xr.DataArray:
    return xr.where(da < 0, 0, da)


def apply_bias(ds_c, method, out, hist_p, _245_p, _585_p):
    dim = 'time'
    method_name = getattr(method, '__name__', 'Unknown')
    print('\t', 'apply', method_name, ': hist', '\r', flush=True, end='')
    basic_qm_hist_chunk = xr.apply_ufunc(method,
                                         ds_c['obs_data'], ds_c['model_data'], ds_c['model_data'],
                                         vectorize=True,
                                         dask='parallelized',
                                         input_core_dims=[[dim], [dim], [dim]],
                                         output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                                         )
    basic_qm_hist = basic_qm_hist_chunk.isel(time=slice(0, 204))
    convert_neg_to_zero(basic_qm_hist.rename('pr')).to_netcdf(save_path(hist_p, out))
    print('\t', 'apply', method_name, ': 245', '\r', flush=True, end='')
    basic_qm_245_chunk = xr.apply_ufunc(method,
                                        ds_c['obs_data'], ds_c['model_data'], ds_c['sce245'],
                                        vectorize=True,
                                        dask='parallelized',
                                        input_core_dims=[[dim], [dim], [dim]],
                                        output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                                        )
    basic_qm_245 = basic_qm_245_chunk.isel(time=slice(204, None))
    convert_neg_to_zero(basic_qm_245.rename('pr')).to_netcdf(save_path(_245_p, out))
    print('\t', 'apply', method_name, ': 585', '\r', flush=True, end='')
    basic_qm_585_chunk = xr.apply_ufunc(method,
                                        ds_c['obs_data'], ds_c['model_data'], ds_c['sce585'],
                                        vectorize=True,
                                        dask='parallelized',
                                        input_core_dims=[[dim], [dim], [dim]],
                                        output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                                        )
    basic_qm_585 = basic_qm_585_chunk.isel(time=slice(204, None))
    convert_neg_to_zero(basic_qm_585.rename('pr')).to_netcdf(save_path(_585_p, out))


def save_path(original_path: Path, out_root: Path):
    time_var = original_path.parent.parent.name
    new_path = out_root / time_var / original_path.name.replace('DECODE', 'BIASED')
    new_path.parent.mkdir(parents=True, exist_ok=True)
    return new_path


# %%
def get_name(name):
    return name.split('_')[4]


out_basic = Path(r'H:\CMIP6 - Biased\pr_basic_qm')
out_mod = Path(r'H:\CMIP6 - Biased\pr_mod_qm')
obs_path = r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc'
pr_attr = {'long_name': 'precipitation', 'units': 'mm/month', 'standard_name': 'precipitation_amount'}
obs = xr.load_dataarray(obs_path)
obs = obs.assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
obs.attrs = pr_attr

hist_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap').iterdir()))
sen245_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\ssp245\decode_cmip_pr_ssp245_2015_2100_noleap').iterdir()))
sen585_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\ssp585\decode_cmip_pr_ssp585_2015_2100_noleap').iterdir()))

for i in range(len(hist_path)):
    ns = get_name(hist_path[i].name), get_name(sen245_path[i].name), get_name(sen585_path[i].name)
    print(ns[0] == ns[1] == ns[2], *ns)


# %%

def open_and_apply_bias(obs_ds, hist_p, sen245_p, sen585_p):
    model = xr.open_dataarray(hist_p)
    sen245 = xr.open_dataarray(sen245_p)
    sen585 = ut.select_year(xr.open_dataarray(sen585_p), 2015, 2100)
    print('\t', 'convert km*m^-2*s^-1 to mm/day ...', '\r', flush=True, end='')
    model_m = ut.km_m2_s__to__mm_day(model).resample(time='MS').sum()
    sen245_m = ut.km_m2_s__to__mm_day(sen245).resample(time='MS').sum()
    sen585_m = ut.km_m2_s__to__mm_day(sen585).resample(time='MS').sum()

    model_m.attrs = pr_attr
    sen245_m.attrs = pr_attr
    sen585_m.attrs = pr_attr
    print('\t', 'merging observe, model, 245, 585 ...', '\r', flush=True, end='')
    # ds = xr.Dataset({'model_data': model_m, 'obs_data': obs_ds, 'sce245': sen245_m, 'sce585': sen585_m})
    ds = xr.merge([obs_ds.rename('obs_data'),
                   model_m.rename('model_data').assign_coords(
                       time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')),
                   sen245_m.rename('sce245').assign_coords(time=pd.date_range('2015-01-01', '2100-12-01', freq='MS')),
                   sen585_m.rename('sce585').assign_coords(time=pd.date_range('2015-01-01', '2100-12-01', freq='MS'))
                   ])
    ds_chunk = ds.chunk({'lat': 16, 'lon': 16})
    # print('\t', 'apply basic quantile:', end='')
    apply_bias(ds_chunk, basic_quantile, out_basic, hist_p=hist_p, _245_p=sen245_p, _585_p=sen585_p)
    # print('\t', 'apply modified quantile:', end='')
    apply_bias(ds_chunk, modified_quantile, out_mod, hist_p=hist_p, _245_p=sen245_p, _585_p=sen585_p)


# %%
skip = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3']
error_list = """
CanESM5

""".strip().split('\n')
error = []
for i in range(len(hist_path)):

    name = get_name(hist_path[i].name)
    if name in error_list:
        s = time.time()
        print(i + 1, get_name(hist_path[i].name))
        try:
            open_and_apply_bias(obs_ds=obs,
                                hist_p=hist_path[i],
                                sen245_p=sen245_path[i],
                                sen585_p=sen585_path[i])
        except Exception as e:
            print('error.')
            error.append([i, name, str(e)])
        e = time.time()
        # else:
        #     print(i + 1, name, 'skipped.')
        print('\tfinished in:', f'{e - s:.2f}')
print('Done')
# %%
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')


def mf_plot2(_mf_ds, level):
    fig, axes = plt.subplots(ncols=3,
                             nrows=4,
                             figsize=(14, 14),
                             subplot_kw={
                                 'projection': ccrs.PlateCarree()
                             },
                             tight_layout=False)
    faxes = axes.flatten()
    # fig.delaxes(faxes[-1])
    # fig.delaxes(faxes[-2])
    for i, ax in enumerate(faxes[:len(_mf_ds.month)]):
        _mf_ds.isel(month=i).plot(ax=ax,
                                  levels=level,
                                  cmap='jet',
                                  )

        ax.set_title(f'month {i + 1}')
        ax.coastlines()
        ax.add_feature(country_borders, edgecolor='darkgray')

    plt.show()
    plt.clf()
    plt.close()


# %%
model = xr.open_dataarray(hist_path[3])
sen245 = xr.open_dataarray(sen245_path[3])
sen585 = xr.open_dataarray(sen585_path[3])


# %%
def mm_day(da, show_time=True):
    s = time.time()
    temp = xr.apply_ufunc(ut.km_m2_s__to__mm_day, da,
                          input_core_dims=[['time', 'lat', 'lon']],
                          output_core_dims=[['time', 'lat', 'lon']],
                          vectorize=True,
                          dask='parallelized'
                          )
    e = time.time()
    if show_time:
        print(e - s)
    return temp


# %%

model_m = mm_day(model).resample(time='MS').sum()
sen245_m = mm_day(sen245).resample(time='MS').sum()
sen585_m = mm_day(sen585).resample(time='MS').sum()

model_m.attrs = pr_attr
sen245_m.attrs = pr_attr
sen585_m.attrs = pr_attr
# %%
# ds = xr.Dataset({'model_data': model_m, 'obs_data': obs, 'sce245': sen245_m, 'sce585': sen585_m})
# print('\t', 'apply basic quantile:', end='')
# apply_bias(ds_chunk, basic_quantile, out_basic, hist_p=hist_p, _245_p=sen245_p, _585_p=sen585_p)
# # print('\t', 'apply modified quantile:', end='')
# apply_bias(ds_chunk, modified_quantile, out_mod, hist_p=hist_p, _245_p=sen245_p, _585_p=sen585_p)
# %%
ds = xr.merge([obs.rename('obs_data'),
               model_m.rename('model_data').assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')),
               sen245_m.rename('sce245').assign_coords(time=pd.date_range('2015-01-01', '2100-12-01', freq='MS')),
               ut.select_year(sen585_m, 2015, 2100).rename('sce585').assign_coords(
                   time=pd.date_range('2015-01-01', '2100-12-01', freq='MS'))
               ])
ds_chunk = ds.chunk({'lat': 16, 'lon': 16})


# %%
def basic_quantile2(obs_data, mod_data, sce_data):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, np.arange(10, 100, 10)) for x in [obs_data, mod_data]])
    return sce_data + cor


def modified_quantile(obs_data, mod_data, sce_data):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])

    mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
    g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
    iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
    iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

    f = np.true_divide(iqr_obs_data, iqr_mod_data)
    cor = g * mid + f * (cor - mid)
    return sce_data + cor


# %%
dim = 'time'
hist_chunk = xr.apply_ufunc(basic_quantile2,
                            ds_chunk['obs_data'], ds_chunk['model_data'], ds_chunk['model_data'],
                            vectorize=True,
                            dask='parallelized',
                            input_core_dims=[[dim], [dim], [dim]],
                            output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                            )
s = time.time()
re = hist_chunk.compute()
e = time.time()
# %%
sample = ds.isel(lat=45, lon=45)
# %%
ob = sample['obs_data'].values
mod = sample['model_data'].values
s245 = sample['sce245'].values
# %%
cdf = ECDF(mod)
p = cdf(s245) * 100
cor = np.subtract(*[np.nanpercentile(x, p) for x in [ob, mod]])
mid = np.subtract(*[np.nanpercentile(x, 50) for x in [ob, mod]])
g = np.true_divide(*[np.nanpercentile(x, 50) for x in [ob, mod]])
iqr_obs_data = np.subtract(*np.nanpercentile(ob, [75, 25]))
iqr_mod_data = np.subtract(*np.nanpercentile(mod, [75, 25]))
f = np.true_divide(iqr_obs_data, iqr_mod_data)
cor = g * mid + f * (cor - mid)
re = s245 + cor
# %%
df = pd.DataFrame()

# %%
np.insert(pd.date_range('1998-01-01', '2014-01-01', freq='MS').strftime('%d-%m-%Y').to_numpy(), 0, ['N', 'E'])
for lat in range(len(obs.lat)):
    for lon in range(len(obs.lon)):
        print(lat, lon, '\r', flush=True, end='')
        ls.append(np.insert(obs.isel(lat=lat, lon=lon).values, 0, [obs.lat[lat].values, obs.lon[lon].values]))
arr = np.array(ls)


# %%
# df = pd.DataFrame(columns=range(len(obs.lat)*len(obs.lon)))
# land_mod = model_m.where(obs.isel(time=0).notnull())


