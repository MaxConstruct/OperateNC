# %%
from pathlib import Path

import xarray as xr
import numpy as np
from util import netcdf_util as ut
import pandas as pd
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import rcParams


# matplotlib setting
# GeoAxes._pcolormesh_patched = Axes.pcolormesh

# Get country border for matplotlib
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
# %%
cru_pr = xr.load_dataarray(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc').assign_coords(
    time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
cru_tmean = xr.load_dataarray(r'H:\Observation\Cleaned Data\tmean\monthly 1998-2014\cru_tmean_monthly_1998_2014.nc').assign_coords(
    time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
pr_hist = sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\historical').iterdir()))
tmean_hist = sorted(list(Path(r'H:\CMIP6 - Biased\tmean\historical').iterdir()))
unbias_pr_hist = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap').iterdir()))
#%%

def get_mfds(paths):
    mf = []
    for i, p in enumerate(paths):
        mf.append(xr.open_dataarray(p).assign_coords(id=i, time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')))
    return xr.concat(mf, dim='id')


mf_tmean = get_mfds(tmean_hist)
mf_pr = get_mfds(pr_hist)
# %%
mme_pr = mf_pr.where(~np.isinf(mf_pr)).mean(dim='id', skipna=True)
mme_tmean = mf_tmean.mean(dim='id', skipna=True)


# %%
def crmsd_func(obs, bias):
    if np.isnan(obs).all():
        return np.nan
    diff_o = obs - np.mean(obs)
    diff_s = bias - np.mean(bias)
    return np.sqrt(np.mean(np.square(diff_s - diff_o)))


def get_cor_rmsd_std(mme, obs):
    std = mme.std(dim='time')
    corre = xr.corr(obs, mme, dim='time')
    rmsd = xr.apply_ufunc(crmsd_func, obs, mme, input_core_dims=[['time'], ['time']], vectorize=True)
    return corre, rmsd, std


# %%

names = ['R', 'RMSD', 'SD']

pr_name = [f'({chr(ord("d")+n)}) {i} (Pmean)' for n, i in enumerate(names)]
tmean_name = [f'({chr(ord("a")+n)}) {i} (Tmean)' for n, i in enumerate(names)]
#%%
dss_pr = dss = get_cor_rmsd_std(mme_pr, cru_pr)
dss_tmean = get_cor_rmsd_std(mme_tmean, cru_tmean)
# %%
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 14
#%%

def plot_cor_rmsd_std(dss, name, level):
    fig, axes = plt.subplots(ncols=3,
                             nrows=1,
                             figsize=(16, 6),
                             subplot_kw={
                                 'projection': ccrs.PlateCarree(),
                             },
                             tight_layout=True)
    faxes = axes.flatten()

    for i, da in enumerate(dss):
        da.plot(ax=faxes[i], cmap='jet', levels=level[i], cbar_kwargs={'shrink': 0.53, 'label':''})
        faxes[i].add_feature(country_borders, edgecolor='darkgray')
        faxes[i].coastlines()
        faxes[i].set_title(name[i])

    plt.subplots_adjust(wspace=0.07, hspace=0)
    plt.show()
    plt.clf()
    plt.close()
#%%
plot_cor_rmsd_std(dss_pr, pr_name, [np.arange(-0.5, 1, 0.01), np.arange(17, 370, 1), np.arange(12, 351, 1)])
plot_cor_rmsd_std(dss_tmean, tmean_name, [np.arange(0, 1, 0.01), np.arange(0, 2, 0.05), np.arange(0, 7, 0.1)])

#%%
def range_max_min(dss):
    l = []
    for i in dss:
        c = np.floor(i.min().values), np.ceil(i.max().values)
        l.append(c)
    return l
#%%
cru_c = cru_pr.mean(dim='time')
mf_pr_unbias = []
for i in unbias_pr_hist:
    print(i.name)
    mf_pr_unbias.append(ut.km_m2_s__to__mm_day(xr.load_dataarray(i)).resample(time='MS').sum(skipna=True))
mf_pr_unbias_cut = [i.where(cru_c.notnull()).assign_coords(id=n) for n, i in enumerate(mf_pr_unbias)]
mf_pr_unbias_cut2 = [i.assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')) for i in mf_pr_unbias_cut]
mf_pr_con = xr.concat(mf_pr_unbias_cut2, dim='id')
mme_pr_unbias = mf_pr_con.where(~np.isinf(mf_pr_unbias)).mean(dim='id', skipna=True)
unbias = get_cor_rmsd_std(mme_pr_unbias, cru_pr)
#%%
plot_cor_rmsd_std(unbias, pr_name, [np.arange(-1, 1, 0.01), np.arange(32, 324, 1), np.arange(31, 228, 1)])
#%%
mf_ds_new = [xr.load_dataarray(p).assign_coords(id=i, time=pd.date_range('1998-01-01', '2014-12-01', freq='MS')) for i, p in enumerate(pr_hist)]
dss_pr_new = [get_cor_rmsd_std(i, cru_pr) for i in mf_ds_new]
corre_new = [i[0] for i in dss_pr_new]
corre__new_mean = xr.concat(corre_new, dim='id').mean(dim='id')