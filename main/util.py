import os
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
# %%
GeoAxes._pcolormesh_patched = Axes.pcolormesh

TEST_ROOT_DIR = 'H:/CMIP6 - Test/'
SEA_MODEL = 'SAMPLE - pr_SEA-25_HadGEM2-AO_historical_r1i1p1_WRF_v3-5_day_197101-198012.nc'
TEST_FILE = 'tasmax_day_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_20141201-20141231.nc'

ROOT_DIR = 'I:/CMIP6'
VARIABLES = ['pr', 'tasmax', 'tasmin']
LABELS = ['hist-1950', 'ssp245', 'ssp585']


def sea_dataset():
    return dataset(SEA_MODEL)


def test_dataset():
    return dataset(TEST_FILE)


def dataset(filename):
    return xr.open_dataset(os.path.join(TEST_ROOT_DIR, filename))


def shift_to_180(ds):
    return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')


def to_celsius(ds, attr):
    ds[attr] = ds[attr] - 273.15
    return ds


def crop_to_model(model, sample):
    new_lat = model['lat']
    new_lon = model['lon']
    return sample.sel(lat=new_lat, lon=new_lon, method='nearest')


def interp_crop_to_model(model, sample):
    new_lat = model['lat']
    new_lon = model['lon']
    return sample.interp(lat=new_lat, lon=new_lon)


def test_path(name):
    return os.path.join(TEST_ROOT_DIR, name)


def plot(dset: xr.DataArray, time=None, savefig=None, show=True, set_global=False, country_border=True, **kwargs):
    ax = plt.axes(projection=ccrs.PlateCarree())

    if time is None:
        time = 0

    dset.isel(time=time).plot(ax=ax, **kwargs)

    if country_border:
        ax.add_feature(country_borders, edgecolor='darkgray')

    if set_global:
        ax.set_global()
    ax.coastlines()

    if savefig is not None:
        plt.savefig(os.path.join(TEST_ROOT_DIR, savefig))

    if show:
        plt.show()
    plt.clf()
    plt.close()
