# %%
from pathlib import Path

import xarray as xr
import numpy as np
from util import netcdf_util as ut

# %%
pr_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_1998_2014_noleap').iterdir()))
tmean_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_tmean_1998_2014_noleap').iterdir()))
cru = xr.load_dataarray(r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc')
# %%
# for p in pr_path:
#     pass
#     ds = xr.load_dataarray(p)
#     ds *= 86400
#     cut_ds = ds.where(cru.notnull())
#     m = cut_ds.resample(time='M').sum(skipna=False)
#     re = m.mean(dim=['lat', 'lon'], skipna=True)
#     print(p.name.split('_')[4], '\t'.join([str(i) for i in re.values]), sep='\t')
#     ds.close()
# %%

tmean_bias_path = sorted(list(Path(r'H:\CMIP6 - Biased\tmean\historical').iterdir()))
pr_basic_qm_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_basic_qm\historical').iterdir()))
pr_mod_qm_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_mod_qm\historical').iterdir()))


# %%
def mean_lat_lon_loop(paths: list, **kwargs):
    for p in paths:
        mean_lat_lon(p, **kwargs)

#%%
def mean_lat_lon(path: Path, name_index=4):
    ds = xr.load_dataarray(path)
    re = ds.where(~np.isinf(ds)).mean(dim=['lat', 'lon'], skipna=True)
    print(path.name.split('_')[name_index], *re.values, sep='\t')
    ds.close()


# %%
# mean_lat_lon_loop(tmean_bias_path)
# %%
# mean_lat_lon_loop(pr_basic_qm_path)
mean_lat_lon_loop(sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\historical').iterdir())), name_index=1)
#%%
