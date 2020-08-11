#%%
import os
import shutil
import numpy as np
from os.path import join, dirname

src = r'I:\CMIP6 - 27 Models'

#%%
def lsdir(src, dname):
    return [join(src, dname, i) for i in os.listdir(join(src, dname))]


#%%
d1 = lsdir(src, 'dump')
d2 = lsdir(src, 'dump2')
d3 = lsdir(src, 'dump3 - firefox')
d4 = lsdir(src, 'dump4 - brave')

dump = np.array(d1 + d2 + d3 + d4)


#%%
def get_move_path(file: str):
    file = os.path.split(file)[-1]
    meta = get_meta(file)
    return join(src, meta[2], meta[0], meta[3], file)


def get_meta(file: str):

    return file[:-3].split('_')


def move_file(_src, _dst):
    os.makedirs(dirname(_dst), exist_ok=True)
    if not os.path.exists(_dst):
        shutil.move(_src, _dst)
    else:
        print('DUPICATE FILE', _dst)

#%%
for f in dump:
    move_file(f, get_move_path(f))
