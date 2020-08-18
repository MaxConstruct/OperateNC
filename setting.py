#%%
from pathlib import Path
from sys import platform

from util.file_util import drive, wsl_path

TEST_ROOT_DIR = drive('h') / 'CMIP6 - Test'
CMIP6_PATH = drive('h') / 'CMIP6 - 27 Models'
CMIP6_SEA = drive('h') / 'CMIP6 - SEA'

SEA_MODEL = TEST_ROOT_DIR / 'SAMPLE - pr_SEA-25_HadGEM2-AO_historical_r1i1p1_WRF_v3-5_day_197101-198012.nc'

#%%
CMIP6_MODEL_NAME = [p.name for p in CMIP6_PATH.iterdir()]

#%%

#%%

