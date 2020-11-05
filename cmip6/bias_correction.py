import warnings

import xarray as xr
import numpy as np
from pathlib import Path
from setting import CMIP6_SEA
from util import netcdf_util as ut
from util import preprocess as pre
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

# %%
def make_12_month_mean(ds):
    return ds.groupby('time.month').mean(dim='time')


def daily_to_monthly_mean(ds):
    return ds.resample(time='M').mean(skipna=False)


def daily_to_monthly_sum(ds):
    return ds.resample(time='M').sum(skipna=False)


def _f(ds, diff=None, operation_func=None):
    return operation_func(ds, diff[ds.time.dt.month[0].values - 1])


def apply_12_to_all_month(ds_all, ds_12, operation_func):
    return ds_all.groupby('time.month').map(_f, diff=ds_12, operation_func=operation_func)


def bounds(d1, d2):
    return np.floor(min(d1.min().values, d2.min().values)), np.ceil(max(d1.max().values, d2.max().values))


def save_path(original_path: Path, out_root: Path):
    time_var = original_path.parent.parent.name
    var = original_path.name.split('_')[2]
    new_path = out_root / var / time_var / original_path.name.replace('DECODE', 'BIASED')
    new_path.parent.mkdir(parents=True, exist_ok=True)
    return new_path


def t_d_func(t_model, observe, t_project=None):
    t_model_mean = make_12_month_mean(t_model)
    obs_mean = make_12_month_mean(observe)
    diff_obs_model = (obs_mean - t_model_mean)
    if t_project is not None:
        return apply_12_to_all_month(ds_all=t_project, ds_12=diff_obs_model, operation_func=lambda x, y: x + y)
    return apply_12_to_all_month(ds_all=t_model, ds_12=diff_obs_model, operation_func=lambda x, y: x + y)


def p_d_func(p_model, observe, p_project=None):
    p_model_mean = make_12_month_mean(p_model)
    obs_mean = make_12_month_mean(observe)
    ratio_obs_model = obs_mean / p_model_mean
    if p_project is not None:
        return apply_12_to_all_month(ds_all=p_project, ds_12=ratio_obs_model, operation_func=lambda x, y: x * y)
    return apply_12_to_all_month(ds_all=p_model, ds_12=ratio_obs_model, operation_func=lambda x, y: x * y)

#%%
def get_bias(model_d, std_ratio):
    model_d_mean = make_12_month_mean(model_d)
    prime_1 = apply_12_to_all_month(ds_all=model_d, ds_12=model_d_mean, operation_func=lambda x, y: x - y)
    prime_2 = apply_12_to_all_month(ds_all=prime_1, ds_12=std_ratio, operation_func=lambda x, y: x * y)
    model_cor_d = apply_12_to_all_month(ds_all=prime_2, ds_12=model_d_mean, operation_func=lambda x, y: x + y)
    return model_cor_d
#%%

def get_name(name):
    return name.split('_')[4]
#%%

observe_tmean = xr.load_dataarray(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014\cru_tmean_monthly_1998_2014.nc')
observe_pr = xr.load_dataarray(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc')

root = Path(r'H:\CMIP6 - SEA\Cleaned')
root_out = Path(r'H:\CMIP6 - Biased')

tmean_path_hist = sorted(list((root / 'historical' / 'decode_cmip_tmean_hist_1998_2014_noleap').iterdir()))
tmean_path_245 = sorted(list((root / 'ssp245' / 'decode_cmip_tmean_ssp245_2015_2100_noleap').iterdir()))
tmean_path_585 = sorted(list((root / 'ssp585' / 'decode_cmip_tmean_ssp585_2015_2100_noleap').iterdir()))

pr_path_hist = sorted(list((root / 'historical' / 'decode_cmip_pr_hist_1998_2014_noleap').iterdir()))
pr_path_245 = sorted(list((root / 'ssp245' / 'decode_cmip_pr_ssp245_2015_2100_noleap').iterdir()))
pr_path_585 = sorted(list((root / 'ssp585' / 'decode_cmip_pr_ssp585_2015_2100_noleap').iterdir()))

if len(tmean_path_hist) == len(tmean_path_245) == len(tmean_path_585):
    print('Pass')
else:
    raise ValueError('len(Path) is not equal')

#%%
# For TMEAN
for i in range(len(tmean_path_hist)):
# for i in [0]:
    print(i+1, 18, get_name(tmean_path_hist[i].name))
    print('\tHistorical')
    ds_hist = daily_to_monthly_mean(xr.open_dataarray(tmean_path_hist[i]))
    t_ref_d = t_d_func(ds_hist, observe_tmean)

    ref_std_ratio = observe_tmean.groupby('time.month').std() / t_ref_d.groupby('time.month').std()

    t_ref_cor_d = get_bias(t_ref_d, ref_std_ratio)
    t_ref_cor_d.to_netcdf(save_path(tmean_path_hist[i], root_out))

    print('\t245')
    ds_245 = daily_to_monthly_mean(xr.open_dataarray(tmean_path_245[i]))
    t_proj_d_245 = t_d_func(ds_hist, observe_tmean, t_project=ds_245)
    t_proj_cor_d_245 = get_bias(t_proj_d_245, ref_std_ratio)
    t_proj_cor_d_245.to_netcdf(save_path(tmean_path_245[i], root_out))
    ds_245.close()

    print('\t585')
    ds_585 = daily_to_monthly_mean(xr.open_dataarray(tmean_path_585[i]))
    t_proj_d_585 = t_d_func(ds_hist, observe_tmean, t_project=ds_585)
    t_proj_cor_d_585 = get_bias(t_proj_d_585, ref_std_ratio)
    t_proj_cor_d_585.to_netcdf(save_path(tmean_path_585[i], root_out))
    ds_585.close()
    ds_hist.close()

print('Done')


#%%
# For PR
# for i in range(len(pr_path_hist)):
i = 0
print(i+1, 18, get_name(pr_path_hist[i].name))
print('\tHistorical')
ds_hist = daily_to_monthly_sum(ut.km_m2_s__to__mm_day(xr.open_dataarray(pr_path_hist[i])))
p_ref_mean = make_12_month_mean(ds_hist)
obs_mean = make_12_month_mean(observe_pr)
p_ref_d = apply_12_to_all_month(ds_all=ds_hist, ds_12=(obs_mean / p_ref_mean), operation_func=lambda x, y: x * y)

ref_std_ratio = observe_pr.groupby('time.month').std() / p_ref_d.groupby('time.month').std()

p_ref_d_mean = make_12_month_mean(p_ref_d)
prime_1 = apply_12_to_all_month(ds_all=p_ref_d, ds_12=p_ref_d_mean, operation_func=lambda x, y: x - y)
prime_2 = apply_12_to_all_month(ds_all=prime_1, ds_12=ref_std_ratio, operation_func=lambda x, y: x * y)
model_cor_d = apply_12_to_all_month(ds_all=prime_2, ds_12=p_ref_d_mean, operation_func=lambda x, y: x + y)

print('Done')


# %%

def plot_diff(before, after, observe, name=''):
    fig, axes = plt.subplots(ncols=2,
                             nrows=2,
                             figsize=(16, 15),
                             subplot_kw={
                                 'projection': ccrs.PlateCarree()
                             },
                             tight_layout=False)
    faxes = axes.flatten()

    min_r, max_r = bounds(before, after)

    before.plot(ax=faxes[0], levels=np.arange(min_r, max_r, 0.01), cmap='jet')
    after.plot(ax=faxes[1], levels=np.arange(min_r, max_r, 0.01), cmap='jet')
    before_diff = observe - before
    after_diff = observe - after

    min_r, max_r = bounds(before_diff, after_diff)
    (observe - after).plot(levels=np.arange(min_r, max_r, 0.01), ax=faxes[3])
    before_diff.plot(levels=np.arange(min_r, max_r, 0.01), ax=faxes[2])

    faxes[0].set_title(f'{name} unbiased')
    faxes[1].set_title(f'{name} biased')
    faxes[2].set_title('different unbiased')
    faxes[3].set_title('different biased')

    for ax in faxes:
        ax.coastlines()

    plt.show()
    plt.clf()
    plt.close()


plot_diff(before=ds_hist[0], after=p_ref_cor_d[0], observe=obs[0])
#%%
import pyperclip
def copy(x):
    pyperclip.copy('\n'.join([str(i) for i in x]))
