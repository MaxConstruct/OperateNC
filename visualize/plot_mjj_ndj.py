from pathlib import Path

import xarray as xr
import numpy as np
from util import netcdf_util as ut
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# %%
# tmean_bias_hist_path = sorted(list(Path(r'H:\CMIP6 - Biased\tmean\historical').iterdir()))
pr_bias_hist_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\historical').iterdir()))
pr_bias_245_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\ssp245').iterdir()))
pr_bias_585_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\ssp585').iterdir()))
cru_path = Path(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc')
cru = xr.load_dataarray(cru_path)

def mean_pr_mjj_ndj(da: xr.DataArray):
    _t = da.isel(time=slice(4, len(da.time) - 2))
    m_5_10 = _t.where((_t.time.dt.month >= 5) & (_t.time.dt.month <= 10), drop=True)
    m_11_4 = _t.where(~((_t.time.dt.month >= 5) & (_t.time.dt.month <= 10)), drop=True)
    m_5_10_mean = m_5_10.groupby('time.month').mean()
    m_11_4_mean = m_11_4.groupby('time.month').mean()

    m_5_10_sum = m_5_10_mean.sum(dim='month', skipna=False)
    m_11_4_sum = m_11_4_mean.sum(dim='month', skipna=False)

    return m_5_10_sum, m_11_4_sum

# %%
mf_hist = ut.get_mfds(pr_bias_hist_path)
mf_245 = ut.get_mfds(pr_bias_245_path)
mf_585 = ut.get_mfds(pr_bias_585_path)

obs, mod = mean_pr_mjj_ndj(cru), mean_pr_mjj_ndj(mf_hist)
near = mean_pr_mjj_ndj(ut.select_year(mf_245, 2015, 2039)), mean_pr_mjj_ndj(ut.select_year(mf_585, 2015, 2039))
mid = mean_pr_mjj_ndj(ut.select_year(mf_245, 2040, 2069)), mean_pr_mjj_ndj(ut.select_year(mf_585, 2040, 2069))
far = mean_pr_mjj_ndj(ut.select_year(mf_245, 2070, 2099)), mean_pr_mjj_ndj(ut.select_year(mf_585, 2070, 2099))

# %%
# mme_hist = mf_hist.mean(dim='id', skipna=True)
# mme_245 = mf_245.mean(dim='id', skipna=True)
# mme_585 = mf_585.mean(dim='id', skipna=True)

# %%
mjj = [float(i[0].mean().values) for i in ([obs, mod] + list(near + mid + far))]
ndj = [float(i[1].mean().values) for i in ([obs, mod] + list(near + mid + far))]

# %%
# df = pd.DataFrame(data=[
#     ['NDJFMA'] + mjj,
#     ['MJJASO'] + ndj
# ],
#     columns=['Time', 'Observation', 'Models', 'Near Future ssp2-4.5', 'Near Future ssp5-8.5', 'Mid Future ssp2-4.5',
#              'Mid Future ssp5-8.5', 'Far Future ssp2-4.5', 'Far Future ssp5-8.5'])
# # %%
# df.set_index('Time').T.plot(kind='bar', stacked=True)
# plt.show()

# %%
df1 = pd.DataFrame(data=[
    ['NDJFMA'] + mjj[:2],
    ['MJJASO'] + ndj[:2]
],
    columns=['Time', 'Observation', 'Model'])
df2 = pd.DataFrame(data=[
    ['NDJFMA'] + mjj[2:4],
    ['MJJASO'] + ndj[2:4]
],
    columns=['Time', 'ssp2-4.5', 'ssp5-8.5'])
df3 = pd.DataFrame(data=[
    ['NDJFMA'] + mjj[4:6],
    ['MJJASO'] + ndj[4:6]
],
    columns=['Time', 'ssp2-4.5', 'ssp5-8.5'])
df4 = pd.DataFrame(data=[
    ['NDJFMA'] + mjj[6:],
    ['MJJASO'] + ndj[6:]
],
    columns=['Time', 'ssp2-4.5', 'ssp5-8.5'])
#%%
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 16

# %%
name = ['Historical', 'Near Future', 'Mid Future', 'Far Future']
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(10, 8))
d = [df1, df2, df3, df4]
for i in range(len(d)):
    d[i].set_index('Time').T.plot(ax=axes[i], kind='bar', stacked=True, legend=False, rot=45, color=['Blue', 'Gold'])
    axes[i].set_title(name[i])
    axes[i].get_xticklabels()[0].set_horizontalalignment('right')
    axes[i].get_xticklabels()[1].set_horizontalalignment('right')
plt.subplots_adjust(wspace=0.0)
# plt.legend(bbox_to_anchor=(0, 1), loc='lower center', ncol=1)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
#%%

