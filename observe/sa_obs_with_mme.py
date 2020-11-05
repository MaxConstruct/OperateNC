import xarray as xr
import numpy as np
import pandas as pd
import util.netcdf_util as ut
from pathlib import Path
from os import startfile as st

#%%

# sao_month = sao.resample(time='MS').sum(skipna=False)
all_obs_path = sorted([Path(i.strip()) for i in
r"""
H:\Observation\Cleaned Data\pr\monthly 1998-2014\gpcc_pr_monthly_1998_2014.nc
H:\Observation\Cleaned Data\pr\monthly 1998-2014\aphrodite_pr_monthly_1998_2014.nc
H:\Observation\Cleaned Data\pr\monthly 1998-2014\cpc_pr_monthly_1998_2014.nc
H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc
""".strip().split('\n')])
# #%%
# all_obs_path = ut.lsdir(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014')
# all_obs_path.pop(-2)
#%%
# obs_all = ut.get_mfds(all_obs_path, check_inf=False)
mf_ds = []
for i, p in enumerate(all_obs_path):
    da = xr.open_dataarray(p).assign_coords(id=i, time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
    print(da.transpose('time', 'lat', 'lon').shape)
    mf_ds.append(da.transpose('time', 'lat', 'lon'))

model = mf_ds[-1]
for i, _ in enumerate(mf_ds):
    if mf_ds[i].shape != (6209, 149, 201):
        print(i)
        # mf_ds[i] = mf_ds[i].interp_like(model)

# #%%
# mme = np.zeros((6209, 149, 201))
# #%%
# mme = np.nanmean(mf_ds, axis=0)
#
# #%%
# obs_mme = xr.DataArray(mme, coords=model.coords)
mf_da = xr.concat(mf_ds, dim='id')
#%%
sa_obs_path = Path(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\sa-obs_pr_monthly_1998_2014.nc')
# obs_mme_path = Path(r'H:\Observation\Cleaned Data\pr\pr_mme_daily_without_sa-obs_1998-2014.nc')
sao = xr.open_dataarray(sa_obs_path).astype(dtype=np.float32)
obs_mme = mf_da.mean(dim='id').astype(dtype=np.float16)
#%%
sao_interp = sao.interp(lat=obs_mme.lat, lon=obs_mme.lon).assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
#%%
def replace_nan(obs_mean, sa):
    # print(obs_mean.shape, sa.shape)
    # print(obs_mean)
    # print(sa)
    # new_sa = np.copy(sa)
    # idx = np.isnan(new_sa)
    # print(idx, new_sa)
    # new_sa[idx] = obs_mean[idx]
    return np.where(np.isnan(sa), obs_mean, sa)
#%%
new = xr.apply_ufunc(replace_nan, obs_mme, sao_interp,
                     input_core_dims=[['time'], ['time']],
                     output_core_dims=[['time']],
                     vectorize=True,
                     )
#%%
ut.plot(sao)
#%%
ut.plot(sao_interp)
#%%
cru = xr.open_dataarray(r'H:\Observation\Cleaned Data\pr\mean annual 1998-2014\cru_pr_annual_1998_2014.nc')
new_cut = new.where(cru.notnull())