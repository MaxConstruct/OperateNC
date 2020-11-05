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

import time

dim = 'time'
# %%
root = Path(r'H:\Test')

obs = xr.open_dataarray(root / 'cru_pr_monthly_1998_2014.nc')
# obs = xr.open_dataarray(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\trmm_pr_1998_2014.nc')

model = xr.open_dataarray(
    r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap\DECODE_SEA_pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_18500101-20141231.nc',
)
sen = xr.open_dataarray(
    r'H:\CMIP6 - SEA\Cleaned\ssp245\decode_cmip_pr_ssp245_2015_2100_noleap\DECODE_SEA_pr_day_ACCESS'
    r'-CM2_ssp245_r1i1p1f1_gn_20150101-21001231.nc',
)

model_m = (model * 86400).resample(time='MS').sum()
model_m.attrs = {'long_name': 'precipitation', 'units': 'mm/month', 'standard_name': 'precipitation_amount'}
# model_m.to_netcdf(r'H:\Test\Test2\access_cm2_hist_m.nc')

sen_m = (sen * 86400).resample(time='MS').sum()
sen_m.attrs = {'long_name': 'precipitation', 'units': 'mm/month', 'standard_name': 'precipitation_amount'}
# rem.to_netcdf(r'H:\Test\Test2\access_cm2_245_m.nc')

# obs = obs.fillna(-999)
obs = obs.assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
obs.attrs = {'long_name': 'precipitation', 'units': 'mm/month', 'standard_name': 'precipitation_amount'}
# obs.to_netcdf(r'H:\Test\Test2\cru_pr_monthly_1998_2014.nc')
ds = xr.Dataset({'model_data': model_m, 'obs_data': obs, 'sce_data': sen_m})
ds_chunk = ds.chunk({'lat': 16, 'lon': 16})
# %%

from bias_correction import XBiasCorrection

# %%
bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
# %%
re2 = bc.correct()

# %%
re3 = bc.correct(method='normal_mapping')


# %%
@numba.jit(nopython=True)
def filter_func(x, k):
    return x >= k


@numba.jit(nopython=True)
def filter2_np_nb(arr, k):
    j = 0
    for i in range(arr.size):
        if filter_func(arr[i], k):
            j += 1
    result = np.empty(j, dtype=arr.dtype)
    j = 0
    for i in range(arr.size):
        if filter_func(arr[i], k):
            result[j] = arr[i]
            j += 1
    return result


@numba.jit(nopython=True)
def where_greater(x, y):
    return np.where(x > y, y, x)


def where_lower(x, y):
    return np.where(x < y, y, x)


@numba.jit(nopython=True)
def np_sort(x):
    return np.sort(x)


@numba.jit(nopython=True)
def interp(x, xp, fp):
    return np.interp(x, xp, fp)


@numba.jit(nopython=True)
def inverse_form(x):
    return 1. / (1. - x)


@numba.jit(nopython=True)
def inverse(obs_cdf_intpol, mod_cdf_intpol, sce_cdf):
    return [inverse_form(x) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]


@numba.jit(nopython=True)
def np_argsort(x):
    return np.argsort(x)


@numba.jit(nopython=True)
def where_con(con, x, y):
    return np.where(con, y, x)


# %%
# @numba.jit(nopython=True)
def gamma_correction(obs_data, mod_data, sce_data, lower_limit=0.1, cdf_threshold=0.9999999):
    temp = [filter2_np_nb(x, lower_limit) for x in [obs_data, mod_data, sce_data]]
    # for i, t in enumerate(temp):
    #     if len(temp[i]) == 0:
    #         temp[i] = np.zeros_like(obs_data)
    obs_raindays, mod_raindays, sce_raindays = temp

    if len(obs_raindays) == 0:
        a = np.empty_like(obs_data)
        a[:] = np.nan
        return a

    obs_gamma, mod_gamma, sce_gamma = [gamma.fit(x) for x in [obs_raindays, mod_raindays, sce_raindays]]
    obs_cdf = gamma.cdf(np_sort(obs_raindays), *obs_gamma)
    mod_cdf = gamma.cdf(np_sort(mod_raindays), *mod_gamma)
    sce_cdf = gamma.cdf(np_sort(sce_raindays), *sce_gamma)

    obs_cdf = where_greater(obs_cdf, cdf_threshold)
    mod_cdf = where_greater(mod_cdf, cdf_threshold)
    sce_cdf = where_greater(sce_cdf, cdf_threshold)

    # obs_cdf = np.where(obs_cdf > cdf_threshold, cdf_threshold, obs_cdf)
    # mod_cdf = np.where(mod_cdf > cdf_threshold, cdf_threshold, mod_cdf)
    # sce_cdf = np.where(sce_cdf > cdf_threshold, cdf_threshold, sce_cdf)

    # obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
    # mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold
    # sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

    obs_cdf_intpol = interp(np.linspace(1, len(obs_raindays), len(sce_raindays)),
                            np.linspace(1, len(obs_raindays), len(obs_raindays)),
                            obs_cdf)

    mod_cdf_intpol = interp(np.linspace(1, len(mod_raindays), len(sce_raindays)),
                            np.linspace(1, len(mod_raindays), len(mod_raindays)),
                            mod_cdf)

    obs_inverse, mod_inverse, sce_inverse = inverse(obs_cdf_intpol, mod_cdf_intpol, sce_cdf)
    # obs_inverse, mod_inverse, sce_inverse = [1. / (1. - x) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]

    adapted_cdf = 1 - 1. / (obs_inverse * sce_inverse / mod_inverse)
    adapted_cdf = where_lower(adapted_cdf, 0)
    # adapted_cdf[adapted_cdf < 0.] = 0.

    initial = gamma.ppf(np_sort(adapted_cdf), *obs_gamma) * gamma.ppf(sce_cdf, *sce_gamma) / gamma.ppf(sce_cdf,
                                                                                                       *mod_gamma)

    # obs_frequency = 1. * obs_raindays.shape[0] / obs_data.shape[0]
    mod_frequency = 1. * mod_raindays.shape[0] / mod_data.shape[0]
    sce_frequency = 1. * sce_raindays.shape[0] / sce_data.shape[0]

    days_min = len(sce_raindays) * sce_frequency / mod_frequency

    expected_sce_raindays = int(min(days_min, len(sce_data)))

    sce_argsort = np_argsort(sce_data)
    correction = np.zeros(len(sce_data))

    if len(sce_raindays) > expected_sce_raindays:
        initial = interp(np.linspace(1, len(sce_raindays), expected_sce_raindays),
                         np.linspace(1, len(sce_raindays), len(sce_raindays)),
                         initial)
    else:
        initial = np.hstack((np.zeros(expected_sce_raindays - len(sce_raindays)), initial))

    correction[sce_argsort[:expected_sce_raindays]] = initial
    # return xr.DataArray()
    # print(correction, type(correction), correction.shape, '\r', flush=True, end='')
    return correction


# %%
corrected = xr.apply_ufunc(gamma_correction,
                           ds_chunk['obs_data'], ds_chunk['model_data'], ds_chunk['sce_data'],
                           vectorize=True, dask='parallelized',
                           input_core_dims=[[dim], [dim], [dim]],
                           output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                           kwargs={'lower_limit': 0.1,
                                   'cdf_threshold': 0.9999999}
                           )
s = time.time()
re = corrected.compute(scheduler='processes')
e = time.time()

print(e - s)


# %%
def detrend_nan(data):
    new_data = data.copy()
    new_data[np.logical_not(pd.isna(new_data))] = detrend(new_data[np.logical_not(pd.isna(new_data))])
    return data


def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    if np.isnan(obs_data).all():
        a = np.empty_like(obs_data)
        a[:] = np.nan
        return a
    # print('obs:', obs_data)
    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    obs_detrended, mod_detrended, sce_detrended = [detrend_nan(x) for x in [obs_data, mod_data, sce_data]]
    # print(obs_detrended)
    obs_norm, mod_norm, sce_norm = [norm.fit(x[~np.isnan(x)]) for x in [obs_detrended, mod_detrended, sce_detrended]]

    obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_diff = sce_data - sce_detrended
    sce_argsort = np.argsort(sce_detrended)

    obs_cdf_intpol = np.interp(np.linspace(1, obs_len, sce_len),
                               np.linspace(1, obs_len, obs_len),
                               obs_cdf)
    mod_cdf_intpol = np.interp(np.linspace(1, mod_len, sce_len),
                               np.linspace(1, mod_len, mod_len),
                               mod_cdf)
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [(x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]

    obs_inverse, mod_inverse, sce_inverse = [1. / (.5 - np.abs(x)) for x in
                                             [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]]

    adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
    adapted_cdf[adapted_cdf < 0] += 1.
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
            norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - sce_mean
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


corrected = xr.apply_ufunc(normal_correction,
                           ds_chunk['obs_data'], ds_chunk['model_data'], ds_chunk['sce_data'],
                           vectorize=True, dask='parallelized',
                           input_core_dims=[[dim], [dim], [dim]],
                           output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                           kwargs={'cdf_threshold': 0.9999999})

s = time.time()
re = corrected.compute()
e = time.time()

print(e - s)


# %%
def quantile_correction(obs_data, mod_data, sce_data, modified=True):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    if modified:
        mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])

        iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
        iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

        f = np.true_divide(iqr_obs_data, iqr_mod_data)
        cor = g * mid + f * (cor - mid)
        return sce_data + cor
    else:
        return sce_data + cor


# %%
corrected = xr.apply_ufunc(quantile_correction,
                           ds_chunk['obs_data'], ds_chunk['model_data'], ds_chunk['sce245'],
                           vectorize=True, dask='parallelized',
                           input_core_dims=[[dim], [dim], [dim]],
                           output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                           kwargs={'modified': True}
                           )

s = time.time()
ref = corrected.compute(scheduler='processes')
e = time.time()

print(e - s)


# %%

def mf_plot(ds, levels=None):
    size = 23
    p = ds.plot(col='month', col_wrap=4, cmap='jet',
                transform=ccrs.PlateCarree(),
                aspect=0.74,
                figsize=(size * 0.658, size),
                # levels=levels,
                cbar_kwargs={
                    'spacing': 'proportional',
                    'orientation': 'horizontal',
                    'shrink': 1,
                    'label': '',
                    'aspect': 40,
                    'anchor': (0.5, 2.0),
                },
                subplot_kws={
                    'projection': ccrs.PlateCarree()
                },
                )
    for i, ax in enumerate(p.axes.flat):
        ax.coastlines()
        ax.add_feature(country_borders, edgecolor='darkgray')
        ax.set_extent([92.5, 142.5, -12.5, 24.5], crs=ccrs.PlateCarree())
        if i < len(ds):
            ax.set_title(str(ds[i].coords['id'].values))
    plt.show()
    plt.clf()
    plt.close()


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
obs_mm = obs.groupby('time.month').mean(skipna=True)
ref_qm_m = re2.groupby('time.month').mean(skipna=True)
max_range = np.ceil(max(obs_mm.max(), ref_qm_m.max()))
diff = (ref_qm_m - obs_mm) / obs_mm * 100
# %%
mf_plot2(obs_mm, np.arange(0, max_range))
mf_plot2(ref_qm_m, np.arange(0, max_range))
mf_plot2(diff, np.arange(-100, 100))


# %%

def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    obs_detrended, mod_detrended, sce_detrended = [detrend(x) for x in [obs_data, mod_data, sce_data]]
    obs_norm, mod_norm, sce_norm = [norm.fit(x) for x in [obs_detrended, mod_detrended, sce_detrended]]

    obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_diff = sce_data - sce_detrended
    sce_argsort = np.argsort(sce_detrended)

    obs_cdf_intpol = np.interp(np.linspace(1, obs_len, sce_len),
                               np.linspace(1, obs_len, obs_len),
                               obs_cdf)
    mod_cdf_intpol = np.interp(np.linspace(1, mod_len, sce_len),
                               np.linspace(1, mod_len, mod_len),
                               mod_cdf)
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [(x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]

    obs_inverse, mod_inverse, sce_inverse = [1. / (.5 - np.abs(x)) for x in
                                             [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]]

    adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
    adapted_cdf[adapted_cdf < 0] += 1.
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
                norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - sce_mean
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


corrected = xr.apply_ufunc(normal_correction,
                           ds_chunk['obs_data'], ds_chunk['model_data'], ds_chunk['sce245'],
                           vectorize=True, dask='parallelized',
                           input_core_dims=[[dim], [dim], [dim]],
                           output_core_dims=[[dim]], output_dtypes=[xr.Dataset],
                           kwargs={'cdf_threshold': 0.9999999})
