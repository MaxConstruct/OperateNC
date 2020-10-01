import xarray as xr
import numpy as np
import pandas as pd
from observe import observe_path as op
import util.netcdf_util as ut
from pathlib import Path
import warnings
from os import startfile as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

daily_p = Path(r'H:\Observation\Cleaned Data\daily 1998-2014')
monthly_p = Path(r'H:\Observation\Cleaned Data\monthly 1998-2014')
annual_p = Path(r'H:\Observation\Cleaned Data\annual mean 1998-2014')

daily_ds_p = sorted([i for i in daily_p.iterdir()])
monthly_ds_p = sorted([i for i in monthly_p.iterdir()])

# %%
attr = {
    'long_name': 'mean annual precipitation',
    'units': 'mm'
}
attr2 = {
    'long_name': 'total precipitation',
    'units': 'mm/day'
}
# %%
attr3 = {
    'long_name': 'total precipitation',
    'units': 'mm/month'
}
# %%
for p in monthly_ds_p[3:]:
    da = xr.open_dataarray(p)
    n_da = da.resample(time='A').sum()
    n_ds_m = n_da.mean(dim='time')
    n_ds_m.attrs = attr
    n_ds_m.to_netcdf(annual_p / p.name.replace('monthly', 'annual'))

# %%
# plotting
pr_ds = sorted([i for i in annual_p.iterdir()])
mf_ds = [xr.open_dataarray(p) for p in pr_ds]
# %%
for cmap in [0]:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(14, 14), subplot_kw={'projection': ccrs.PlateCarree()}, tight_layout=False)
    faxes = axes.flatten()
    # fig.suptitle(cmap.title())
    fig.delaxes(faxes[-1])
    fig.delaxes(faxes[-2])
    for i, ax in enumerate(faxes[:len(mf_ds)]):
        mf_ds[i].plot(ax=ax, levels=np.arange(0, 8200))
        ax.coastlines()
        ax.set_title(pr_ds[i].name.split('_')[0].title())
    # plt.savefig(ut.test_path('plot1.png'))
    plt.show()
    plt.clf()
    plt.close()
# %%
fig = plt.figure(figsize=(8, 10), tight_layout=False)

columns = 3
rows = 3
projex = ccrs.PlateCarree()


def plotmymap(axs, ds, name):
    plims = ds.plot(ax=axs, levels=np.arange(0, 8000), label=None, add_colorbar=False)
    axs.coastlines()
    axs.set_title(name)
    return plims


for i in range(1, len(mf_ds)+1):
    # add a subplot into the array of plots
    ax = fig.add_subplot(rows, columns, i, projection=projex)
    plims = plotmymap(ax, mf_ds[i-1], pr_ds[i-1])  # a simple maps is created on subplot

# add a subplot for vertical colorbar
bottom, top = 0.1, 0.9
left, right = 0.1, 0.8
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
cbar_ax = fig.add_axes([0.85, bottom, 0.05, top - bottom])
fig.colorbar(plims, cax=cbar_ax)  # plot colorbar

plt.show()  # this plot all the maps
#%%
# Plot solo

for i, ds in enumerate(mf_ds[:1]):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds.plot(ax=ax, levels=np.arange(0, 8000), cmap='cool')
    ax.set_title(pr_ds[i].name.split('_')[0].title())
    ax.coastlines()
    plt.show()
    plt.clf()
    plt.close()
#%%


