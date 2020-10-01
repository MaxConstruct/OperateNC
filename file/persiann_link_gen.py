#%%
import requests
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs


def req_str(year):
    return f'https://www.ncei.noaa.gov/data/precipitation-persiann/access/{year}/'


req_strs = [req_str(y) for y in range(1998, 2014 + 1)]
ls = []
for c, req in enumerate(req_strs):
    print(c, req)
    re = requests.get(req)
    be = bs(re.content.decode('UTF-8'))
    ls += [urljoin(req, i.get('href')) for i in be.find_all('a') if i.get('href').startswith('PERSIANN-CDR')]

with open(r'C:\Users\DEEP\PycharmProjects\NCFile\file\persiann_links.txt', mode='w') as file:
    file.write('\n'.join(ls))

print('Done.')