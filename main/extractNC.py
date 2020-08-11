# import netCDF4 as nc
#
# tasmax_path_file = r"H:\CMIP6\tasmax\hist-1950\CMCC-CM2-HR4\tasmax_day_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_20141201-20141231.nc"
# destdir = r'H:\CMIP6\Test\out.nc'
#
# with nc.Dataset(tasmax_path_file) as src, nc.Dataset(destdir, "w") as dst:
#     # copy global attributes all at once via dictionary
#
#     dst.setncatts(src.__dict__)
#     # copy dimensions
#     for name, dimension in src.dimensions.items():
#         dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
#
#     # copy all file data except for the excluded
#     for name, variable in src.variables.items():
#         x = dst.createVariable(name, variable.datatype, variable.dimensions)
#         dst[name][:] = src[name][:]
#         # copy variable attributes all at once via dictionary
#         dst[name].setncatts(src[name].__dict__)