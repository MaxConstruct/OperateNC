
from pathlib import Path
from setting import CMIP6_SEA
import xarray as xr
import numpy as np
import util.netcdf_util as ut
from os import startfile as st
def model_name(file_name):
    return file_name.split('_')[4]

tmean_attr = {
    'standard_name': 'air_temperature',
    'long_name': 'Mean Near-Surface Air Temperature',
    'units': 'Celsius'
}

clean = Path(r'H:\CMIP6 - SEA\Cleaned')
#%%
for path in ['ssp245', 'ssp585']:

    root = clean / path

    tasmax_paths = sorted(list((root / f'decode_cmip_tasmax_{path}_2015_2100_noleap').iterdir()))
    tasmin_paths = sorted(list((root / f'decode_cmip_tasmin_{path}_2015_2100_noleap').iterdir()))

    out = root / f'decode_cmip_tmean_{path}_2015_2100_noleap'

    if not out.exists():
        out.mkdir()

    for i in range(18):

        tmax_name = model_name(tasmax_paths[i].name)
        tmin_name = model_name(tasmin_paths[i].name)

        if tmax_name != tmin_name:
            raise ValueError('Model is not match.')

        print(i + 1, 18, tmax_name)

        new_name = f'DECODE_SEA_tmean_day_{tmax_name}_{path}_20150101-21001231.nc'

        print(i+1, path, new_name)

        tmax = xr.open_dataarray(tasmax_paths[i])
        tmin = xr.open_dataarray(tasmin_paths[i])

        re = (tmax + tmin) * 0.5 - 273.15
        re.attrs = tmean_attr
        re = re.rename('tmean')

        re.to_netcdf(out / new_name)



        tmax.close()
        tmin.close()
    print('------')
print('Done.')
