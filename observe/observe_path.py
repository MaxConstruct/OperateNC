
from pathlib import Path

#%%

aph = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] Aphrodite 1901 & 1808\APHRO_MA_025deg_V1901_1998-2015_Prep.nc',
    'tmean': r'H:\Observation\Raw Data (SEA)\[SEA] Aphrodite 1901 & 1808\APHRO_MA_TAVE_025deg_V1808_1961-2015_Temp.nc'
}
cpc = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] CPC-UNI\CNC_UNI_prep_1979-2019.nc'
}
era = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] ERA-Interim\ei_tp_sea_025km_1979_2018.nc',
    'tmin': r'H:\Observation\Raw Data (SEA)\[SEA] ERA-Interim\ei_mn2t_sea_025km_1979_2018.nc',
    'tmax': r'H:\Observation\Raw Data (SEA)\[SEA] ERA-Interim\ei_mx2t_sea_025km_1979_2018.nc'
}
gpc = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] GPCP_1DD_v1.2\GPCP_1DD_v1.2_199610-201510.nc'
}
jra = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] JRA-55\fcst_phy2m.061_tprat.reg_tl319.1998010100_2018123121.nc',
    'tmax': r'H:\Observation\Raw Data (SEA)\[SEA] JRA-55\minmax_surf.016_tmax.reg_tl319.1998010100_2018123121.nc',
    'tmin': r'H:\Observation\Raw Data (SEA)\[SEA] JRA-55\minmax_surf.016_tmin.reg_tl319.1998010100_2018123121.nc'

}
trm = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] TRMM_3B42_Daily (TMPA)\TRMM_3B42_Daily_1998-2019.nc'
}
cru = {
    'pr': r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.pre.dat.nc',
    'tmean': r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.tmp.dat.nc',
    'tmin': r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.tmn.dat.nc',
    'tmax': r'H:\Observation\Raw Data (SEA)\[SEA] CRU_TS_4.04\cru_ts4.04.1901.2019.tmx.dat.nc'
}
#%%
o_all = [aph, cpc, era, gpc, jra, trm, cru]
o_pr = [Path(i['pr']) for i in o_all if 'pr' in i.keys()]
o_tmax = [Path(i['tmax']) for i in o_all if 'tmax' in i.keys()]
o_tmin = [Path(i['tmin']) for i in o_all if 'tmin' in i.keys()]
o_tmean = [Path(i['tmean']) for i in o_all if 'tmean' in i.keys()]

