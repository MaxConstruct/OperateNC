from scipy.stats import gamma
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from io import StringIO
import pandas as pd
import util.netcdf_util as ut
import ray
import xarray as xr
from pathlib import Path

ray.init(include_dashboard=True)


# %%
def sorted_values(Obs, Sim):
    count = 0
    for i in range(len(Obs)):
        if Obs[i] == 0:
            count += 1
    Rank = [i + 1 for i in range(len(Obs))]
    Dict = dict(zip(Rank, Sim))
    # SortedSim = sorted(Dict.values())
    SortedSim = sorted(Dict.values())
    # SortedRank = sorted(Dict, key=Dict.get)
    SortedRank = sorted(Dict, key=Dict.get)

    for i in range(count):
        SortedSim[i] = 0
    ArrangedDict = dict(zip(SortedRank, SortedSim))
    SortedDict_by_Rank = sorted(ArrangedDict.items())
    # SortedDict_by_Rank = np.sort(ArrangedDict.items())

    ArrangedSim = [v for k, v in SortedDict_by_Rank]
    return ArrangedSim


def sorted_values_thresh(sim, fut):
    try:
        # Min_Positive_Value_Sim = min(i for i in sim if i > 0)
        Min_Positive_Value_Sim = min(i for i in sim if i > 0)

    except:
        Min_Positive_Value_Sim = 0
    for i in range(len(fut)):
        if fut[i] < Min_Positive_Value_Sim:
            fut[i] = 0
    return fut


def to_xarray(df: pd.DataFrame, coords):
    dset = xr.DataArray(coords=coords)
    for i in range(1, len(df.columns)):
        lat, lon = df[i][:2]
        dset.loc[dict(lat=lat, lon=lon)] = df[i][2:].astype(np.float32)
    return dset


# %%
def format_csv(observe_csv_path, model_csv_path, future_csv_path):
    ObsHData, ModHData, ModFData = [], [], []
    with open(observe_csv_path) as f:
        line = [line for line in f]
        for i in range(len(line)):
            ObsHData.append([word for word in line[i].split(",") if word])

    with open(model_csv_path) as f:
        line = [line for line in f]
        for i in range(len(line)):
            ModHData.append([word for word in line[i].split(",") if word])

    with open(future_csv_path) as f:
        line = [line for line in f]
        for i in range(len(line)):
            ModFData.append([word for word in line[i].split(",") if word])

    return ObsHData, ModHData, ModFData


@ray.remote
def gamma_rain_bias(ObsHData, ModHData, ModFData):

    CorrectedData = []

    lat = [float(ObsHData[0][c]) for c in range(1, len(ObsHData[0]))]
    lon = [float(ObsHData[1][c]) for c in range(1, len(ObsHData[0]))]

    # DateObsH = [ObsHData[r][0] for r in range(len(ObsHData))]
    DateModH = [ModHData[r][0] for r in range(len(ModHData))]
    DateModF = [ModFData[r][0] for r in range(len(ModFData))]

    CorrectedData.append(DateModF)

    # YObsH = int(DateObsH[2][-4:])
    YModH = int(DateModH[2][-4:])
    YModF = int(DateModF[2][-4:])
    for j in range(len(lat)):

        ObsH = np.array([float(ObsHData[r][j + 1]) for r in range(2, len(ObsHData))], dtype=np.float32)
        ModH = np.array([float(ModHData[r][j + 1]) for r in range(2, len(ModHData))], dtype=np.float32)
        ModF = np.array([float(ModFData[r][j + 1]) for r in range(2, len(ModFData))], dtype=np.float32)
        # DateObsH = [date(YObsH, 1, 1) + timedelta(i) for i in range(len(ObsH))]
        DateModH = [date(YModH, 1, 1) + relativedelta(months=i) for i in range(len(ModH))]
        DateModF = [date(YModF, 1, 1) + relativedelta(months=i) for i in range(len(ModF))]

        """
        Start Bias
        """

        DateH = DateModH
        DateF = DateModF
        ModH_Month = []
        ModF_Month = []
        Cor_Monthwise = []
        ObsH_Monthwise = [[] for m in range(12)]
        ObsH_MonthFreq = [[] for m in range(12)]
        ModH_Monthwise = [[] for m in range(12)]
        ModH_MonthFreq = [[] for m in range(12)]
        ModF_Monthwise = [[] for m in range(12)]
        ModF_MonthFreq = [[] for m in range(12)]
        DateH_Monthwise = [[] for m in range(12)]
        DateF_Monthwise = [[] for m in range(12)]

        for m in range(12):
            for i in range(len(ObsH)):
                if DateH[i].month == m + 1:
                    DateH_Monthwise[m].append(DateH[i])
                    ObsH_Monthwise[m].append(ObsH[i])
                    ModH_Monthwise[m].append(ModH[i])
        for m in range(12):
            for i in range(len(ModF)):
                if DateF[i].month == m + 1:
                    DateF_Monthwise[m].append(DateF[i])
                    ModF_Monthwise[m].append(ModF[i])
        for m in range(12):
            ModH_Month.append(sorted_values(ObsH_Monthwise[m], ModH_Monthwise[m]))
            ModF_Month.append(sorted_values_thresh(ModH_Month[m], ModF_Monthwise[m]))

        ModH_Monthwise = ModH_Month
        ModF_Monthwise = ModF_Month

        for m in range(12):
            for i in range(len(ModH_Monthwise[m])):
                if ModH_Monthwise[m][i] > 0:
                    ModH_MonthFreq[m].append(ModH_Monthwise[m][i])
                if ObsH_Monthwise[m][i] > 0:
                    ObsH_MonthFreq[m].append(ObsH_Monthwise[m][i])
            for i in range(len(ModF_Monthwise[m])):
                if ModF_Monthwise[m][i] > 0:
                    ModF_MonthFreq[m].append(ModF_Monthwise[m][i])

        for m in range(12):
            Cor = []

            if len(ModH_MonthFreq[m]) > 0 and len(ObsH_MonthFreq[m]) > 0 and len(ModF_MonthFreq[m]) > 0:
                Moh, Mgh, Mgf, Voh, Vgh, Vgf = np.mean(ObsH_MonthFreq[m]), np.mean(ModH_MonthFreq[m]), np.mean(
                    ModF_MonthFreq[m]), np.std(ObsH_MonthFreq[m]) ** 2, np.std(ModH_MonthFreq[m]) ** 2, np.std(
                    ModF_MonthFreq[m]) ** 2

                if not any(param < 0.000001 for param in [Moh, Mgh, Mgf, Voh, Vgh, Vgf]):
                    aoh, boh, agh, bgh, agf, bgf = Moh ** 2 / Voh, Voh / Moh, Mgh ** 2 / Vgh, Vgh / Mgh, Mgf ** 2 / Vgf, Vgf / Mgf
                    # loh, lgh, lgf = 0, 0, 0
                else:
                    aoh, loh, boh = gamma.fit(ObsH_MonthFreq[m], loc=0)
                    agh, lgh, bgh = gamma.fit(ModH_MonthFreq[m], loc=0)
                    # agf, lgf, bgf = gamma.fit(ModF_MonthFreq[m], loc=0)
                'CDF of ModF with ModH Parameters'
                Prob_ModF_ParaModH = gamma.cdf(ModF_Monthwise[m], agh, scale=bgh)

                'Inverse of Prob_ModF_ParaModH with ParaObsH to get corrected transformed values of Future Model Time Series'
                Cor = gamma.ppf(Prob_ModF_ParaModH, aoh, scale=boh)

            else:

                for i in range(len(ModF_Monthwise[m])):
                    Cor.append(0)

            for c in Cor:
                Cor_Monthwise.append('%.1f' % c)

        """
        End Bias
        """
        Date_Month = []
        for m in range(12):
            for i in range(len(DateF_Monthwise[m])):
                Date_Month.append(DateF_Monthwise[m][i])
        DateCorr_Dict = dict(zip(Date_Month, Cor_Monthwise))

        SortedCorr = sorted(DateCorr_Dict.items())
        CorrectedData.append([lat[j], lon[j]] + [v for k, v in SortedCorr])

    csv_ls = [(','.join(str(CorrectedData[r][c]) for r in range(len(CorrectedData)))) for c in
              range(len(CorrectedData[0]))]
    csv_io = StringIO('\n'.join(csv_ls))
    return pd.read_csv(csv_io, header=None)



# %%
obs = r'H:\CMIP6 - SEA\csv\pr\new_sa_obs.csv'
coor_nc = ut.select_year(xr.open_dataset(r'H:\CMIP6 - Test\ssp_coords_2015-01-01_2100-12-01.nc'), 2015, 2099)
hist = ut.lsdir(r'H:\CMIP6 - SEA\csv\pr\historical')
ssp245 = ut.lsdir(r'H:\CMIP6 - SEA\csv\pr\ssp245')
ssp585 = ut.lsdir(r'H:\CMIP6 - SEA\csv\pr\ssp585')
out_nc = Path(r'H:\CMIP6 - Biased\pr_gamma\nc')
out_csv = Path(r'H:\CMIP6 - Biased\pr_gamma\csv')


# %%
@ray.remote
def run(ssp_p):
    for i, h in enumerate(hist):
        time_var = ssp_p[i].parent.name
        name = ssp_p[i].name
        print(i, time_var, name)
        modf_17 = ut.lsdir(ssp_p[i])
        r = [gamma_rain_bias.remote(*format_csv(obs, h, f)) for f in modf_17]
        re = ray.get(r)
        con = pd.concat([re[0]] + [i.loc[2:] for i in re[1:]])
        con.to_csv(ut.save_file(out_csv / time_var / f'Biased_{name}_2015_2099.csv'))
        con_nc = to_xarray(con, coords=coor_nc.coords)
        con_nc.to_netcdf(ut.save_file(out_nc / time_var / f'Biased_{name}_2015_2099.nc'))


if __name__ == '__main__':
    w = [run.remote(ssp) for ssp in [ssp245, ssp585]]
    ray.get(w)
