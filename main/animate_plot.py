from matplotlib.axes import Axes
import matplotlib.animation as anim

from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import main.util as util
import os
import xarray as xr
import cartopy.feature as cfeat
import animatplot as amp

GeoAxes._pcolormesh_patched = Axes.pcolormesh

# %%
ds = xr.open_dataset(util.test_path('pr_test7_gzip_com5_h5.nc'))


# %%
def make_figure():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # generate a basemap with country borders, oceans and coastlines
    ax.add_feature(cfeat.LAND)
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.COASTLINE)
    ax.add_feature(cfeat.BORDERS, linestyle='dotted')
    return fig, ax


# %%
temp = ds.pr
fig, ax = make_figure()

frames = temp.time.size        # Number of frames
min_value = temp.values.min()  # Lowest value
max_value = temp.values.max()  # Highest value


def draw(frame, add_colorbar):
    grid = temp[frame]
    contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(),
                        add_colorbar=add_colorbar, vmin=min_value, vmax=max_value)
    title = u"%s — %s" % ('PR', str(temp.time[frame].values)[:19])
    ax.set_title(title)
    return contour


def init():
    return draw(0, add_colorbar=True)


def animate(frame):
    return draw(frame, add_colorbar=False)

#%%
ani = anim.FuncAnimation(fig, animate, frames, interval=0.01, blit=False,
                              init_func=init, repeat=False)

#%%
writer = anim.FFMpegFileWriter(fps=25, metadata=dict(artist='jir'), bitrate=1800)
ani.save(util.test_path('pr.mp4'), writer=writer)
plt.close(fig)
