import pandas as pd

#%%
def req_str(datetime):
    return f'https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg/{datetime.strftime("%Y")}/{datetime.strftime("%m")}/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{datetime.strftime("%Y%m%d")}.nc'

dt = pd.date_range('1998-01-01', '2014-12-31')

ls = [req_str(date) for date in dt]

with open(r'C:\Users\DEEP\PycharmProjects\NCFile\file\CMORPH_link.txt', mode='w') as file:
    file.write('\n'.join(ls))

