import geopandas as gs
import rioxarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapefile as shp

# %%

# file = gs.read_file(r'C:\Users\DEEP\Desktop\cliped\raster_th_2_FeatureTo3DByAtt.shp')


# %%

#
map = shp.Reader(r'I:\coastaldem\crop_th\THA_Adm1_GISTA_plyg_v5.shp')
# m = gs.read_file(r'I:\coastaldem\crop_th\THA_Adm1_GISTA_plyg_v5.shp')
# %%
# ax = plt
ext = [99.5927918, 101.24787816, 12.47583841, 14.78633896]
# %%
# for shape in map.shapeRecords():
#     x = [i[0] if (i[0] >= 99.5927918) and (i[0] <= 101.24787816) else np.nan for i in shape.shape.points[:]]
#     y = [i[1] if (i[0] >= 12.47583841) and (i[0] <= 14.78633896) else np.nan for i in shape.shape.points[:]]
#     plt.plot(x, y, 'g')

# %%
# r = xr.open_rasterio(r'I:\coastaldem\crop_th\cp_rt_31.tif')
r = xr.open_rasterio(r'I:\coastaldem\crop_th\crop_extends41.tif')
# %%
im = r[0]
im = im.drop('band')
# %%
# %%
# ax = plt.axes(projection=ccrs.PlateCarree())
# m.plot(ax=ax, edgecolor="gray", facecolor="None", alpha=0.5)
# %%
# im.where((im <= 1.95) & (im >= 0)).plot(
#     ax=ax,
#     levels=np.arange(-1, 10, 1),
#     cmap='Blues_r'
# )
# %%
s1 = [1.95, 2.45, 1.92, 3.07]
s2 = [2.00, 2.73, 2.09, 4.04]
# %%
s = s2
fig, axes = plt.subplots(ncols=2, nrows=2,
                       figsize=(14, 14),
                       # subplot_kw={'projection': ccrs.PlateCarree(),
                       #             }
                       )

for i, ax in enumerate(axes.flatten()):
    tmp_im = s[i] - im
    im.where(im <= 0).plot(
        ax=ax,
        levels=None,
        cmap='Blues',
        label=None,
        add_colorbar=False
        # ax=ax,
        # levels=np.arange(-1, 10, 1),
        # cmap='Blues_r'
    )
    tmp_im.where((tmp_im >= 0) & ((s[i] - tmp_im) > 0)).plot(
        ax=ax,
        levels=np.arange(-1, 4.5, 0.25),
        cmap='winter_r'
    )

    ax.set_title(f'{s[i]} Meter')
plt.savefig(r'H:/CMIP6 - Test/Scenario 2 - Sea Levels.png')
plt.show()
# %%
# im.where((im <= 1.95) & (im >= 0)).plot(
#     ax=ax[0][0],
#     levels=np.arange(-1, 10, 1),
#     cmap='Blues_r'
# )
# ax[0][0].set_title('1.95 Meter')
#
# im.where((im <= 2.45) & (im >= 0)).plot(
#     ax=ax[0][1],
#     levels=np.arange(-1, 10, 1),
#     cmap='Blues_r'
# )
# ax[0][1].set_title('2.45 Meter')
#
# im.where((im <= 1.92) & (im >= 0)).plot(
#     ax=ax[1][0],
#     levels=np.arange(-1, 10, 1),
#     cmap='Blues_r'
# )
# ax[1][0].set_title('1.92 Meter')
#
# im.where((im <= 3.07) & (im >= 0)).plot(
#     ax=ax[1][1],
#     levels=np.arange(-1, 10, 1),
#     cmap='Blues_r'
# )
# ax[1][1].set_title('3.07 Meter')

# fig.suptitle('Scenario 1')

plt.savefig(r'H:/CMIP6 - Test/scenario 1.png')
plt.show()

# %%
# for shape in map.shapeRecords()[-2: -1]:
#     x = [i[0] for i in shape.shape.points[:]]
#     y = [i[1] for i in shape.shape.points[:]]
#     print(x)
#     # plt.plot(x, y, 'g')
