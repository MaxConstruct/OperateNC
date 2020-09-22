import util.netcdf_util as ut

sea_dataset = ut.sea_dataset()
new_lon = sea_dataset['lon']
new_lat = sea_dataset['lat']
lat_bnds = [-12, 24.5]
lon_bnds = [92.5, 142.5]


def crop_sea(data, x_name='lat', y_name='lon'):
    return ut.crop_dataset_from_bound(data, lon_bound=lon_bnds, lat_bound=lat_bnds, x_name=x_name, y_name=y_name)


# %%
def regrid_sea(data, x_name='lat', y_name='lon', crop=False):

    if not crop:
        return data.interp(lat=new_lat, lon=new_lon)

    mask_lon = ut.select_range(data[y_name], lon_bnds[0], lon_bnds[1])
    mask_lat = ut.select_range(data[x_name], lat_bnds[0], lat_bnds[1])

    n_ds = data.isel(lat=mask_lat, lon=mask_lon, drop=True)

    return n_ds.interp(lat=new_lat, lon=new_lon)
