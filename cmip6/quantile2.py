#%%
#!/usr/bin/python
"""programme to make quantile(CDF)-mapping bias correction"""
"""Adrian Tompkins tompkins@ictp.it - please feel free to use"""
import numpy as np


def map(vals):
    """ CDF mapping for bias correction """
    """ note that values exceeding the range of the training set"""
    """ are set to -999 at the moment - possibly could leave unchanged?"""
    # calculate exact CDF values using linear interpolation
    #
    cdf1 = np.interp(vals, xbins, cdfmod, left=0.0, right=999.0)
    # now use interpol again to invert the obsCDF, hence reversed x,y
    corrected = np.interp(cdf1, cdfobs, xbins, left=0.0, right=-999.0)
    return corrected


# ----------------
# MAIN CODE START
# ----------------
#%%
n = 100
cdfn = 10
# make some fake observations(obs) and model (mod) data:
obs = np.random.uniform(low=0.5, high=13.3, size=(n,))
mod = np.random.uniform(low=1.4, high=19.3, size=(n,))
# sort the arrays
obs = np.sort(obs)
mod = np.sort(mod)
# calculate the global max and bins.
global_max = max(np.amax(obs), np.amax(mod))
wide = global_max / cdfn
xbins = np.arange(0.0, global_max + wide, wide)
# create PDF
pdfobs, _ = np.histogram(obs, bins=xbins)
pdfmod, _ = np.histogram(mod, bins=xbins)
# create CDF with zero in first entry.
cdfobs = np.insert(np.cumsum(pdfobs), 0, 0.0)
cdfmod = np.insert(np.cumsum(pdfmod), 0, 0.0)
# dummy model data list to be bias corrected
raindata = [2.0, 5.0]
print(raindata)
print(map(raindata))

#%%
from statsmodels.distributions.empirical_distribution import ECDF

cdf = ECDF(obs)
cdf([2.0, 5.0])