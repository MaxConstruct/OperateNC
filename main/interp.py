#%%
# Import libraries and set configuration
import xarray as xr
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from nco import nco
import pyproj

GeoAxes._pcolormesh_patched = Axes.pcolormesh
#%%

def shift_to_180(ds):
    """Shift coordinate from (0, 360) to (-180, 180)"""
    return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')


def kelvin_to_celsius(ds, attr):
    """Convert kelvin to celsius deg"""
    ds[attr] = ds[attr] - 273.15
    return ds


def crop_to_model(sample, model):
    """Crop model matching sample model coordinate"""
    new_lat = sample['lat']
    new_lon = sample['lon']
    return model.sel(lat=new_lat, lon=new_lon, method='nearest')


def interp_crop_to_model(sample, model):
    """Crop model matching sample model both coordinate and resolution"""
    new_lat = sample['lat']
    new_lon = sample['lon']
    return model.interp(lat=new_lat, lon=new_lon)

#%%
def plot(dset: xr.DataArray, time=None, savefig=None, show=True, set_global=False, projection=None):

    """Plot graph of data array"""
    if projection is None:
        projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    if time is None:
        time = 0
    dset.isel(time=time).plot(ax=ax)
    if set_global:
        ax.set_global()
    ax.coastlines()

    if savefig is not None:
        plt.savefig(savefig)

    if show:
        plt.show()
    plt.clf()
    plt.close()

#%%
# Configuration
FILE_PATH = r'I:\Cordex-EAS-SEA\EAS\CCLM5\EC-EARTH\pr\hist\pr_EAS-44_ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CCLM5-0-2_v1_day_19510101-19551231.nc'

SAMPLE_PATH = r'I:\CMIP6 - Test\SAMPLE - pr_SEA-25_HadGEM2-AO_historical_r1i1p1_WRF_v3-5_day_197101-198012.nc'
OUTPUT_PATH = r'output_path'



SEA = r'I:\Cordex-EAS-SEA\SEA\RegCM4-3\EC-EARTH\pr\hist\pr_SEA-22_ICHEC-EC-EARTH_historical_r1i1p1_ICTP-RegCM4-3_v4_day_1970010112-1970013112.nc'
# Open dataset
sds = xr.open_dataset(SAMPLE_PATH)
ds = xr.open_dataset(FILE_PATH)
#%%
# Shift dataset coordinate to range (-180, 180)
# new_dataset = shift_to_180(dataset)

# Interpolate dataset to match sample file. both coordinate and resolution.
new_lat = sds['lat']
new_lon = sds['lon']
#%%
nds = interp_crop_to_model(sds, ds)

#Plot Dataset

#%%
plot(ds['pr'], set_global=True, projection=ccrs.PlateCarree())
#%%
plot(ds['pr'], set_global=False, projection=ccrs.RotatedPole(pole_latitude=77.61, pole_longitude=-64.78))
#%%
plot(nds['pr'], set_global=False, projection=ccrs.RotatedPole(pole_latitude=77.61, pole_longitude=-64.78))

#%%
ds.rio.reproject('+proj=ob_tran +o_proj=latlon +o_lon_p=0.0 +o_lat_p=77.61 +lon_0=115.22 +to_meter=0.017453292519943295')













