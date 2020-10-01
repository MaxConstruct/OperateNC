# %%
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

# %%
r_path = Path(r'H:\Observation\Raw Data (SEA)\[SEA] TRMM_3B42_Daily (TMPA)')
paths = sorted([i for i in r_path.iterdir()])
out = Path(r'H:\Observation\Cleaned Data\pr\trmm_pr_daily_1998_2019.nc')


# %%
mfd_arr = []
drop_var = ['randomError_cnt', 'randomError', 'precipitation_cnt', 'IRprecipitation', 'IRprecipitation_cnt',
            'HQprecipitation', 'HQprecipitation_cnt']

base = xr.open_dataarray(paths[0])
coords = base.coords
attr = base.attrs
# %%
sm = []
len_paths = len(paths)
for i, p in enumerate(paths):
    da = xr.load_dataarray(p)
    print(f'\r{i + 1}/{len_paths} : {p.name}', end='', flush=True)
    sm.append(da.values.T)
    da.close()

print()
print('To numpy...')
nsm = np.array(sm)
print('Done.')
# %%
mfd = xr.DataArray(
    nsm,
    {
        'time': pd.date_range('1998-01-01', '2019-12-30'),
        'lat': coords['lat'].values,
        'lon': coords['lon'].values
    },
    dims=['time', 'lat', 'lon'],
    attrs=attr,
    name='pr'
)
