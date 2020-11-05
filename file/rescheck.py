import os
from pathlib import Path

import numpy as np
import xarray as xr
from setting import CMIP6_PATH

# %%

root = CMIP6_PATH
skip = """
CESM2
CESM2-WACCM
CNRM-CM6-1-HR
EC-Earth3-Veg
FGOALS-f3-L
MPI-ESM1-2-HR
UKESM1-0-LL
""".split()


def paths(p: Path):
    return sorted(list(p.iterdir()))

#%%
def rescheck(ds, name=''):
    print(name, f"{ds['lon'].diff(dim='lon').values[0]:.2f}×{ds['lat'].diff(dim='lat').values[0]:.2f}", sep='\t')

#%%
def check(name):
    models = paths(Path(r'H:\CMIP6 - 27 Models') / name)
    print(name)
    for ex in models:
        print('\t',ex.name)
        for time in paths(ex):
            ds = xr.open_dataset(paths(time)[0])
            rescheck(ds)
            ds.close()
