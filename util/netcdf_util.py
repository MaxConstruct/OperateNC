# Import libraries and set configuration

# os used for path and directory management
import os

# xarray, is the most important library, used for manipulate netCDF Dataset operation
import xarray as xr
import numpy as np

# matplotlib for plotting Dataset. cartopy for various map projection
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# matplotlib setting
GeoAxes._pcolormesh_patched = Axes.pcolormesh

# %%
# Get country border for matplotlib
country_borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
# %%
# Configuration and setting for project.
TEST_ROOT_DIR = r'H:\CMIP6 - Test'
SEA_MODEL = r'SAMPLE - pr_SEA-25_HadGEM2-AO_historical_r1i1p1_WRF_v3-5_day_197101-198012.nc'


def listdir_abs(path, condition=None):
    """
    :param path: path of directory
    :param condition: condition to filter names in directory
    :return: List of files in the directory with absolute path.
    """
    if condition is None:
        return [os.path.join(path, i) for i in os.listdir(path)]
    else:
        return [os.path.join(path, i) for i in os.listdir(path) if condition(i)]


def sea_dataset():
    """
    Get SEA sample dataset
    :return: xarray Dataset
    """
    return xr.open_dataset(os.path.join(TEST_ROOT_DIR, SEA_MODEL))


def shift_to_180(data):
    """
    Shift coordinate from (0, 360) to (-180, 180)
    :param data: xarray Dataset
    :return: Shifted xarray dataset
    """
    return data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon')


def kelvin_to_celsius(data, attr_name):
    """

    :param data: xrray Dataset
    :param attr_name: variable name for converting, e.g. 'tasmax' or 'tasmin'
    :return: converted dataset
    """
    data[attr_name] = data[attr_name] - 273.15
    return data


def new_coord_array(lon_bound, lat_bound, res):
    """
    Get new lat lot coordinate as xarray DataArray
    :param lon_bound: list of [min_lon, max_lon]
    :param lat_bound: list of [min_lat, max_lat]
    :param res: resolution for
    :return:
    """

    d_lon = np.arange(lon_bound[0], lon_bound[1], res)
    d_lat = np.arange(lat_bound[0], lat_bound[1], res)

    _lon = xr.DataArray(d_lon, coords=[('lon', d_lon)], dims=['lon'])
    _lat = xr.DataArray(d_lat, coords=[('lat', d_lat)], dims=['lat'])

    return _lon, _lat


def select_range(data, _min, _max):
    """
    create boolean array for selecting data within specific range.
    min <= data <= max
    :param data: DataArray to be selected.
    :param _min: min value
    :param _max: max value
    :return: boolean DataArray
    """
    return (data >= _min) & (data <= _max)


def crop_dataset_from_bound(data, lon_bound, lat_bound):
    """
    Crop dataset in to specific lat & lon boundary
    :param data: xarray.Dataset to be cropped
    :param lon_bound: list that contain [min_lon, max_lon]
    :param lat_bound: list that contain [min_lat, max_lat]
    :return: cropped dataset as xarray.Dataset
    """

    mask_lon = select_range(data['lon'], lon_bound[0], lon_bound[1])
    mask_lat = select_range(data['lat'], lat_bound[0], lat_bound[1])

    return data.isel(lat=mask_lat, lon=mask_lon, drop=True)


def crop_dataset_like(model, sample):
    """
    Crop model matching sample model coordinate
    :param model: Dataset to be cropped
    :param sample: target Dataset to matching
    :return: Cropped dataset as xarray.Dataset
    """
    new_lat = sample['lat']
    new_lon = sample['lon']

    return model.sel(lat=new_lat, lon=new_lon, method='nearest', drop=True)


def interpolate_like(model, sample):
    """
    Cropping and regriding model matching sample dataset
    :param model: Dataset to be interpolate
    :param sample: target Dataset to matching
    :return: Interpolated dataset
    """
    new_lat = sample['lat']
    new_lon = sample['lon']
    return model.interp(lat=new_lat, lon=new_lon)


def test_path(name):
    """
    Get absolute path to testing directory

    Example: test_path(foo.txt) will return 'H:/CMIP6 - Test/foo.txt'
    :param name: name to join to test directory
    :return: absolute path
    """
    return os.path.join(TEST_ROOT_DIR, name)


def plot(data: xr.DataArray, time=None, savefig=None, show=True, set_global=False, country_border=True, **kwargs):
    """
    Quick plotting DataArray using PlateCarree as a projection
    Example: plot(dataset['tasmax'])

    :param data: DataArray to be plotted
    :param time: Specific time index to be plot using xarray.DataArray.isec method.
    Default is 0.
    :param savefig: path of figure to being save. Default is None (not save figure).
    :param show: Is showing graphic plot. Default is True.
    :param set_global: Set plotting to show global map.
    :param country_border: Is show country border on the plot. Default is True.
    :param kwargs: keyword arg pass to xarray.DataArray.plot
    :return: None
    """
    ax = plt.axes(projection=ccrs.PlateCarree())

    if time is None:
        time = 0

    data.isel(time=time).plot(ax=ax, **kwargs)

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
