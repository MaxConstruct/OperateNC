import xarray as xr
import numpy as np
import matplotlib.pyplot as plt




# map = shp.Reader(r'I:\coastaldem\crop_th\THA_Adm1_GISTA_plyg_v5.shp')
# ext = [99.5927918, 101.24787816, 12.47583841, 14.78633896]

# Load Raster file
r = xr.open_rasterio(r'I:\coastaldem\crop_th\crop_extends41.tif')
# %%
im = r[0]
im = im.drop('band')

# %%
# Scenario 1
s1 = [1.95, 2.45, 1.92, 3.07]

# Scenario 2
s2 = [2.00, 2.73, 2.09, 4.04]
# %%
# Plot ground levels
s = s2
fig, axes = plt.subplots(ncols=2, nrows=2,
                         figsize=(14, 14),
                         )

for i, ax in enumerate(axes.flatten()):
    im.where(im <= 0).plot(
        ax=ax,
        levels=None,
        cmap='Blues',
        label=None,
        add_colorbar=False
    )

    im.where((im <= s[i]) & (im > 0)).plot(
        ax=ax,
        levels=np.arange(0, 4.25, 0.25),
        cmap='copper_r'
    )
    ax.set_title(f'{s[i]} Meter')
plt.savefig(r'H:/CMIP6 - Test/Scenario 2 - Ground Levels.png')
plt.show()


# %%
# Plot sea levels

s = s2
fig, axes = plt.subplots(ncols=2, nrows=2,
                         figsize=(14, 14),
                         # subplot_kw={'projection': ccrs.PlateCarree(),
                         #             }
                         )

for i, ax in enumerate(axes.flatten()):
    tmp_im = -im + s[i]
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
