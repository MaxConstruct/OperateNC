from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np
import util.netcdf_util as ut

out = Path(r'H:\CMIP6 - SEA\csv\pr')


# %%
def format_cdbc(da, date_col=None):
    df = pd.DataFrame()
    if date_col is None:
        date_col = np.insert(da.time.dt.strftime('%d-%m-%Y').values, 0, ['N', 'E'])
    df[0] = date_col
    i = 1
    for lat in range(len(da.lat)):
        for lon in range(len(da.lon)):
            tmp = da.isel(lat=lat, lon=lon)
            if not np.isnan(tmp.values).all():
                print('\tconverting', lat, lon, '\r', flush=True, end='')
                df[i] = np.insert(tmp, 0, [da.lat[lat].values, da.lon[lon].values])
            i += 1
    return df


def to_xarray(df: pd.DataFrame, coords):
    dset = xr.DataArray(coords=coords)
    for i in range(1, len(df.columns)):
        print(i, '\r', flush=True, end='')
        lat, lon = df[i][:2]
        dset.loc[dict(lat=lat, lon=lon)] = df[i][2:].astype(np.float32)
    return dset


# format_cdbc(model_m).to_csv('mod_with_null.csv', index=False, header=False)

def get_name(name):
    return name.split('_')[4]


# obs_path = r'H:\Observation\Cleaned Data\pr\monthly 1998-2014\cru_pr_monthly_1998_2014.nc'
pr_attr = {'long_name': 'precipitation', 'units': 'mm/month', 'standard_name': 'precipitation_amount'}
obs_path = Path(r'H:\CMIP6 - Test\new_sa_obs_land_monthly_1998-2014.nc')
#%%
obs = xr.load_dataarray(obs_path)
obs = obs.assign_coords(time=pd.date_range('1998-01-01', '2014-12-01', freq='MS'))
#%%
obs.attrs = pr_attr
obs_land = obs.mean(dim='time')
hist_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\historical\decode_cmip_pr_hist_1998_2014_noleap').iterdir()))
sce245_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\ssp245\decode_cmip_pr_ssp245_2015_2100_noleap').iterdir()))
sce585_path = sorted(list(Path(r'H:\CMIP6 - SEA\Cleaned\ssp585\decode_cmip_pr_ssp585_2015_2100_noleap').iterdir()))

for i in range(len(hist_path)):
    ns = get_name(hist_path[i].name), get_name(sce245_path[i].name), get_name(sce585_path[i].name)
    print(ns[0] == ns[1] == ns[2], *ns)
#%%
format_cdbc(obs).to_csv(out/'new_sa_obs.csv', index=False, header=False)
# %%
def open_and_format(paths: list):
    l = len(paths)
    for i, path in enumerate(paths):
        print(i + 1, l, path.name)
        da = xr.load_dataarray(path).resample(time='MS').sum()
        da_m = da.where(obs_land.notnull())
        time_var = path.parent.parent.name
        folder_name = path.name.split('_')[4]
        for i in range(2015, 2100, 17):
            r = ut.select_year(da_m, i, i+16)
            out_path = ut.save_file(out / time_var / folder_name / f'{folder_name}_{time_var}_{i}-{i+16}.csv')
            print('\tsaving', out_path, '\r', flush=True, end='')
            format_cdbc(r).to_csv(out_path, index=False, header=False)
        print()
#%%

da = xr.load_dataarray(sce245_path[0]).resample(time='MS').sum()
format_cdbc(ut.select_year(da, 2032, 2047).where(obs_land.notnull())).to_csv(ut.test_path('ssp245_2032-2047.csv'), index=False, header=False)
#%%
# print('Hist')
# print('------------------')
# open_and_format(hist_path)
print('ssp245')
print('------------------')
open_and_format(sce245_path)
print('ssp585')
print('------------------')
open_and_format(sce585_path)
#%%w
ls = [Path(r'H:\CMIP6 - Biased\pr_gamma\csv\ssp245'), Path(r'H:\CMIP6 - Biased\pr_gamma\csv\ssp585')]
sample_coor = xr.load_dataarray(sce245_path[0])
sample_coor_m = sample_coor.resample(time='MS').sum()

out_nc = Path(r'H:\CMIP6 - Biased\pr_gamma\nc')
#%%
for l in ls:
    bias_p = sorted(list(l.iterdir()))
    print(l.name)
    print('------------')
    for path in bias_p:
        print(path.name)
        cor = pd.read_csv(list(Path(path).iterdir())[0], header=None)
        # print(cor.shape)
        cor_nc = to_xarray(cor, sample_coor_m.coords)
        cor_nc.to_netcdf(ut.save_file(out_nc / l.name / ('Biased_' + path.name + '_2015_2100.nc')))

# open_and_format([sce585_path[3], sce585_path[-3]])
#%%
nc_path = sorted(list(Path(r'H:\CMIP6 - Biased\pr_gamma\nc\ssp245').iterdir()))
mf_ds_arr = [xr.open_dataarray(p).assign_coords(id=i+1) for i, p in enumerate(nc_path)]
mf_ds = xr.concat(mf_ds_arr, dim='id')
#%%
ut.sim_plot(mf_ds.isel(time=0), col='id', col_wrap=4, add_colorbar=False)
#%%
near_p = Path(r'H:\CMIP6 - Test\cdbc\new\Bias Corrected Rainfall 2015.csv')
cor = pd.read_csv(near_p, header=None)
cor_nc2 = to_xarray(cor, ut.select_year(sample_coor_m, 2015, 2031).coords)
