# Import libraries and set configuration

# os used for path and directory management
import os

# xarray, is the most important library, used for manipulate netCDF Dataset operation
import xarray as xr
import numpy as np

# matplotlib for plotting Dataset. cartopy for various map projection
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from setting import SEA_MODEL, TEST_ROOT_DIR

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

def sea_dataset():
    """
    Get SEA sample dataset
    :return: xarray Dataset
    """
    return xr.open_dataset(SEA_MODEL)


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


def new_coord_array(lon_bound, lat_bound, res, x_name='lon', y_name='lat'):
    """
    Get new lat lot coordinate as xarray DataArray
    :param lon_bound: list of [min_lon, max_lon]
    :param lat_bound: list of [min_lat, max_lat]
    :param res: resolution for
    :return:
    """

    d_lon = np.arange(lon_bound[0], lon_bound[1], res)
    d_lat = np.arange(lat_bound[0], lat_bound[1], res)

    _lon = xr.DataArray(d_lon, coords=[(x_name, d_lon)], dims=[x_name])
    _lat = xr.DataArray(d_lat, coords=[(y_name, d_lat)], dims=[y_name])

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


def crop_dataset_from_bound(data, lon_bound, lat_bound, x_name='lon', y_name='lat'):
    """
    Crop dataset in to specific lat & lon boundary
    :param data: xarray.Dataset to be cropped
    :param lon_bound: list that contain [min_lon, max_lon]
    :param lat_bound: list that contain [min_lat, max_lat]
    :return: cropped dataset as xarray.Dataset
    """

    mask_lon = select_range(data[x_name], lon_bound[0], lon_bound[1])
    mask_lat = select_range(data[y_name], lat_bound[0], lat_bound[1])

    return data.isel(lat=mask_lat, lon=mask_lon, drop=False)


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


def test_path(name='.'):
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
        if 'time' in data.coords and data.time.shape != ():
            data.isel(time=0).plot(ax=ax, **kwargs)
        else:
            data.plot(ax=ax, **kwargs)
    else:
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


def sim_plot(ds, **kwargs):
    """
    Simple plotting graph without any setting for matplotlib.
    :param ds: DataArray to plot
    :param kwargs: argument pass to DataArray.plot()
    :return: void
    """
    ds.plot(**kwargs)
    plt.show()
    plt.clf()
    plt.close()



def merge_regrid(paths, out_dst, preprocess, _open_option=None, _save_option=None):
    """
    Open multiple dataset from list of paths, then, merging and regridding
    and save processing file as netcdf.

    :param preprocess: Preprocess option
    :param paths: list of paths of dataset to open and merge
    :param out_dst: path to save processed dataset
    :param _open_option: dictionary option pass to xarray.open_mfdatset
    :param _save_option: dictionary option pass to xarray.Dataset.to_netcdf
    :return: Status of the operation as tuple (boolean, message)
    If operation is success return (True, 'success') otherwise, (False, error message)
    """

    if _save_option is None:
        _save_option = {}
    if _open_option is None:
        _open_option = {}

    try:
        with xr.open_mfdataset(paths=paths,
                               preprocess=preprocess,
                               parallel=True,
                               **_open_option) as mf_dataset:
            os.makedirs(os.path.dirname(out_dst), exist_ok=True)
            mf_dataset.to_netcdf(out_dst, **_save_option)

    except Exception as ex:
        print('\t\t', f"{bcolors.FAIL}Error {bcolors.ENDC} {str(ex)}")
        return False, str(ex)

    return True, 'success'


def time_range(_paths):
    """
    Get time range from given list of paths
    example:
        [foo_1908-1909.nc, foo_1910-1911.nc, ..., foo_2019-2020.nc]
        will return: '1908-2020'

    :param _paths: list contains names
    :return: New name with lowest time range to max
    """
    t0 = str(_paths[0]).split('_')[-1].split('-')[0]
    t1 = str(_paths[-1]).split('_')[-1].split('-')[-1].replace('.nc', '')
    return '{}-{}'.format(t0, t1)


def select_year(ds, from_y, to_y):
    return ds.sel(time=ds.time.dt.year.isin(np.arange(from_y, to_y + 1)))

