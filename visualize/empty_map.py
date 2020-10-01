#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gs
import rioxarray
import xarray as xr
from distributed.deploy.old_ssh import bcolors
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapefile as shp
#%%
# use low resolution coastlines.
map = Basemap(lat_0=45, lon_0=-100, resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='coral', lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

# contour data over the map.
plt.show()
#%%
