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

#%%
clean_dir = Path(r'H:\Observation\Cleaned Data')
pr_path = clean_dir / 'pr' / 'annual mean 1998-2014'
tasmax_path = clean_dir / 'tasmax' / 'annual mean 1998-2014'
tasmin_path = clean_dir / 'tasmin' / 'annual mean 1998-2014'
tmean_path = clean_dir / 'tmean' / 'annual mean 1998-2014'
#%%
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