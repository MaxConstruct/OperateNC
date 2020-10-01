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

# %%
clean_dir = Path(r'H:\Observation\Cleaned Data')
pr_path = clean_dir / 'pr' / 'mean annual 1998-2014'
tasmax_path = clean_dir / 'tasmax' / 'mean annual 1998-2014'
tasmin_path = clean_dir / 'tasmin' / 'mean annual 1998-2014'
tmean_path = clean_dir / 'tmean' / 'mean annual 1998-2014'
plot_path = Path(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img')


# %%

def mean_path(mf_ds, name):
    fig, axes = plt.subplots(ncols=4, nrows=3, subplot_kw={'projection': ccrs.PlateCarree()},
                             tight_layout=False)
    faxes = axes.flatten()
    fig.suptitle(name)
    fig.delaxes(faxes[-1])
    # fig.delaxes(faxes[-2])
    for i, ax in enumerate(faxes[:len(mf_ds)]):
        mf_ds[i].plot(ax=ax,
                      # levels=np.arange(10, 32),
                      levels=np.arange(0, 8500),
                      label=None)
        ax.coastlines()
        ax.set_title(ds_path[i].name.split('_')[0].title())
    plt.savefig(plot_path / f'{name}.png')
    plt.show()
    plt.clf()
    plt.close()
# mean_path(tmean_path, 'Mean annual temperature (Celsius) 1998-2014')
mean_path(pr_path, 'Mean annual precipitation (mm) 1998-2014')
# %%
ds_path = sorted([i for i in pr_path.iterdir()])
mf_ds = [xr.open_dataarray(p) for p in ds_path]
# %%

new_mf = [d.assign_coords(id=ds_path[i].name.split('_')[0].upper())
          for i, d in enumerate(mf_ds)
          ]
all_ds = xr.concat(new_mf, dim='id')
#%%
mx=all_ds.plot(
        x="lon",
        y="lat",
        col="id",
        col_wrap=4,
        add_colorbar=False,
    )
#%%
# cbar = plt.colorbar(mx)
# mx.fig.colorbar(mx.axes[-1, -1], cax=cbar_ax, orientation="horizontal")
plt.show()
plt.clf()
plt.close()

