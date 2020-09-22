# %%
import numpy as np
from pathlib import Path
import requests
import pandas as pd
# %%
raw_path = r'C:\Users\DEEP\PycharmProjects\NCFile\file\cmip5_all.txt'
with open(raw_path) as rf:
    ls = np.array([i.strip() for i in rf.readlines()])


# %%

def struct(link_meta: list):
    return link_meta[0].split('_') + link_meta[1:]


# %%

def subtract_link(file_content: str):
    spl = file_content.split('EOF--dataset.file.url.chksum_type.chksum')
    return [struct(i.split()) for i in spl[1].replace("'", "").split('\n') if len(i.strip()) != 0]


# %%

model_names = [i.strip() for i in
"""
ACCESS1.0
ACCESS1.3
CMCC-CESM
CMCC-CM
CMCC-CMS
CNRM-CM5
CSIRO-Mk3.6.0
CanESM2
EC-EARTH
FGOALS-g2
FGOALS-s2
GFDL-CM3
GFDL-ESM2G
GFDL-ESM2M
HadGEM2-AO
HadGEM2-CC
HadGEM2-ES
INM-CM4
IPSL-CM5A-LR 
IPSL-CM5A-MR
IPSL-CM5B-LR
MIROC-ESM
MIROC-ESM-CHEM
MIROC5
MPI-ESM-LR
MPI-ESM-MR
MRI-CGCM3
NorESM1-M
""".split('\n') if len(i.strip()) > 0]
#%%
selected_model = [i.strip() for i in
"""
ACCESS1.0
ACCESS1.3
CMCC-CESM
CMCC-CM
CMCC-CMS
CNRM-CM5
CSIRO-Mk3.6.0
CanESM2
EC-EARTH
FGOALS-g2
FGOALS-s2
GFDL-CM3
GFDL-ESM2G
GFDL-ESM2M
HadGEM2-AO
HadGEM2-CC
HadGEM2-ES
INM-CM4
IPSL-CM5A-LR 
IPSL-CM5A-MR
IPSL-CM5B-LR
MIROC-ESM
MIROC-ESM-CHEM
MIROC5
MPI-ESM-LR
MPI-ESM-MR
MRI-CGCM3
NorESM1-M
""".split('\n') if len(i.strip()) > 0]

#%%


def req_str(model_name):
    return f'http://esgf-node.llnl.gov/esg-search/wget?limit=10000&latest=true&experiment=historical,rcp45,rcp85&time_frequency=day&ensemble=r1i1p1&variable=pr,tasmax,tasmin&project=CMIP5&model={model_name}'


# %%
links = []
for name in model_names:
    print(name)
    re = requests.get(req_str(name))
    links += subtract_link(re.content.decode('UTF-8'))
link_arr = np.array(links)
print('Done.')
