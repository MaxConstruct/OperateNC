from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gs
import shapefile as shp
from matplotlib.collections import PatchCollection
from shapely.geometry.polygon import LinearRing, Polygon
import util.netcdf_util as ut
# font = {
#     'family': 'normal',
#     'weight': 'bold',
#     'size': 22}
#%%
path_out = Path(r'H:\Scenario Ploting')
plt.rcParams.update({'font.size': 28})
# %%

# ext = [99.5927918, 101.24787816, 12.47583841, 14.78633896]
#%%
# Load Raster file
r = xr.open_rasterio(r'I:\coastaldem\crop_th\crop_extends41.tif')
im = r[0]
im = im.drop('band')
#%%
# Load Bangkok polygon
map = shp.Reader(r'I:\coastaldem\crop_th\THA_Adm1_GISTA_plyg_v5.shp')
bkk = np.array(map.shapeRecords()[8].shape.points)

#%%
ring = LinearRing(bkk)
x, y = ring.xy
# %%
# Plot ground levels
plt.figure(figsize=(20, 20))
ax = plt.axes()
# Sea
im.where(im <= 0).plot(
    ax=ax,
    levels=None,
    cmap='Blues',
    label=None,
    add_colorbar=False
)

# Land
im.where(im > 0).plot(
    ax=ax,

    levels=np.concatenate([np.arange(0, 4.25, 0.25), [1e6]]),
    cmap='copper_r'
)

# Plot Bangkok
ax.plot(x, y, color='red', linewidth=3)

# Save image
plt.title('Ground Levels - 4.25 Meter')
plt.savefig(path_out / 'Ground levels.png')
plt.show()


# %%


# %%
# Plot sea levels
def plot_sea_levels(scenario, out, ncols=2, nrows=2, figsize=(14, 14)):
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        tmp_im = -im + scenario[i]
        im.where(im <= 0).plot(
            ax=ax,
            levels=None,
            cmap='Blues',
            label=None,
            add_colorbar=False
        )
        tmp_im.where((tmp_im >= 0) & ((scenario[i] - tmp_im) > 0)).plot(
            ax=ax,
            levels=np.arange(-1, 4.5, 0.25),
            cmap='winter_r'
        )
        ax.plot(x, y, color='red', linewidth=2)

        ax.set_title(f'{scenario[i]} Meter')
    plt.savefig(out)


s = [
    [1.92, 2.11, 2.75, 3.46],
    [2.02, 2.21, 2.85, 3.56],
    [2.63, 2.82, 3.46, 4.17]
]
# %%
path_out = Path(r'H:\Scenario Ploting\with_bkk_map')
#%%
for i, se in enumerate(s[:1]):
    # plot_ground_levels(se, path_out / f'Scenario {i+1} - Ground Levels.png')
    plot_sea_levels(se, path_out / f'Scenario {i + 1} - Sea Levels.png', figsize=(20, 20))
