import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

df = pd.DataFrame()
df = pd.read_csv('exclude_null.csv')

df['date']=pd.to_datetime(df['date'], infer_datetime_format=True)
 
df.info()   

df = df.reset_index(drop = True)

loca=df['location'].unique()
# print(loca)
all_area=df.copy()

gng=all_area[all_area['location']=='강남구']
gng1 = pd.DataFrame(gng)
print(gng)

Q1_gng = gng1[['d_pm10']].quantile(q=0.25)

Q3_gng = gng1[['d_pm10']].quantile(q=0.75)

IQR_gng = Q3_gng-Q1_gng

IQR_df_gng = gng1[(gng1['d_pm10'] <= Q3_gng['d_pm10']+1.5*IQR_gng['d_pm10']) & (gng1['d_pm10'] >= Q1_gng['d_pm10']-1.5*IQR_gng['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gdg=all_area[all_area['location']=='강동구']
gdg1 = pd.DataFrame(gdg)

Q1_gdg = gdg1[['d_pm10']].quantile(q=0.25)

Q3_gdg = gdg1[['d_pm10']].quantile(q=0.75)

IQR_gdg = Q3_gdg-Q1_gdg

IQR_df_gdg = gdg1[(gdg1['d_pm10'] <= Q3_gdg['d_pm10']+1.5*IQR_gdg['d_pm10']) & (gdg1['d_pm10'] >= Q1_gdg['d_pm10']-1.5*IQR_gdg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gbg= all_area[all_area['location']=='강북구']
gbg1 = pd.DataFrame(gbg)

Q1_gbg = gbg1[['d_pm10']].quantile(q=0.25)

Q3_gbg = gbg1[['d_pm10']].quantile(q=0.75)

IQR_gbg = Q3_gbg-Q1_gbg

IQR_df_gbg = gbg1[(gbg1['d_pm10'] <= Q3_gbg['d_pm10']+1.5*IQR_gbg['d_pm10']) & (gbg1['d_pm10'] >= Q1_gbg['d_pm10']-1.5*IQR_gbg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gsg= all_area[all_area['location']=='강서구']
gsg1 = pd.DataFrame(gsg)

Q1_gsg = gsg1[['d_pm10']].quantile(q=0.25)

Q3_gsg = gsg1[['d_pm10']].quantile(q=0.75)

IQR_gsg = Q3_gsg-Q1_gsg

IQR_df_gsg = gsg1[(gsg1['d_pm10'] <= Q3_gsg['d_pm10']+1.5*IQR_gsg['d_pm10']) & (gsg1['d_pm10'] >= Q1_gsg['d_pm10']-1.5*IQR_gsg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gag= all_area[all_area['location']=='관악구']
gag1 = pd.DataFrame(gag)

Q1_gag = gag1[['d_pm10']].quantile(q=0.25)

Q3_gag = gag1[['d_pm10']].quantile(q=0.75)

IQR_gag = Q3_gag-Q1_gag

IQR_df_gag = gag1[(gag1['d_pm10'] <= Q3_gag['d_pm10']+1.5*IQR_gag['d_pm10']) & (gag1['d_pm10'] >= Q1_gag['d_pm10']-1.5*IQR_gag['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gjg= all_area[all_area['location']=='광진구']
gjg1 = pd.DataFrame(gjg)

Q1_gjg = gjg1[['d_pm10']].quantile(q=0.25)

Q3_gjg = gjg1[['d_pm10']].quantile(q=0.75)

IQR_gjg = Q3_gjg-Q1_gjg

IQR_df_gjg = gjg1[(gjg1['d_pm10'] <= Q3_gjg['d_pm10']+1.5*IQR_gjg['d_pm10']) & (gjg1['d_pm10'] >= Q1_gjg['d_pm10']-1.5*IQR_gjg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

grg= all_area[all_area['location']=='구로구']
grg1 = pd.DataFrame(grg)

Q1_grg = grg1[['d_pm10']].quantile(q=0.25)

Q3_grg = grg1[['d_pm10']].quantile(q=0.75)

IQR_grg = Q3_grg-Q1_grg

IQR_df_grg = grg1[(grg1['d_pm10'] <= Q3_grg['d_pm10']+1.5*IQR_grg['d_pm10']) & (grg1['d_pm10'] >= Q1_grg['d_pm10']-1.5*IQR_grg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

gcg= all_area[all_area['location']=='금천구']
gcg1 = pd.DataFrame(gcg)

Q1_gcg = gcg1[['d_pm10']].quantile(q=0.25)

Q3_gcg = gcg1[['d_pm10']].quantile(q=0.75)

IQR_gcg = Q3_gcg-Q1_gcg

IQR_df_gcg = gcg1[(gcg1['d_pm10'] <= Q3_gcg['d_pm10']+1.5*IQR_gcg['d_pm10']) & (gcg1['d_pm10'] >= Q1_gcg['d_pm10']-1.5*IQR_gcg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

nwg= all_area[all_area['location']=='노원구']
nwg1 = pd.DataFrame(nwg)

Q1_nwg = nwg1[['d_pm10']].quantile(q=0.25)

Q3_nwg = nwg1[['d_pm10']].quantile(q=0.75)

IQR_nwg = Q3_nwg-Q1_nwg

IQR_df_nwg = nwg1[(nwg1['d_pm10'] <= Q3_nwg['d_pm10']+1.5*IQR_nwg['d_pm10']) & (nwg1['d_pm10'] >= Q1_nwg['d_pm10']-1.5*IQR_nwg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

dbg= all_area[all_area['location']=='도봉구']
dbg1 = pd.DataFrame(dbg)

Q1_dbg = dbg1[['d_pm10']].quantile(q=0.25)

Q3_dbg = dbg1[['d_pm10']].quantile(q=0.75)

IQR_dbg = Q3_dbg-Q1_dbg

IQR_df_dbg = dbg1[(dbg1['d_pm10'] <= Q3_dbg['d_pm10']+1.5*IQR_dbg['d_pm10']) & (dbg1['d_pm10'] >= Q1_dbg['d_pm10']-1.5*IQR_dbg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

ddmg= all_area[all_area['location']=='동대문구']
ddmg1 = pd.DataFrame(ddmg)

Q1_ddmg = ddmg1[['d_pm10']].quantile(q=0.25)

Q3_ddmg = ddmg1[['d_pm10']].quantile(q=0.75)

IQR_ddmg = Q3_ddmg-Q1_ddmg

IQR_df_ddmg = ddmg1[(ddmg1['d_pm10'] <= Q3_ddmg['d_pm10']+1.5*IQR_ddmg['d_pm10']) & (ddmg1['d_pm10'] >= Q1_ddmg['d_pm10']-1.5*IQR_ddmg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

djg= all_area[all_area['location']=='동작구']
djg1 = pd.DataFrame(djg)

Q1_djg = djg1[['d_pm10']].quantile(q=0.25)

Q3_djg = djg1[['d_pm10']].quantile(q=0.75)

IQR_djg = Q3_djg-Q1_djg

IQR_df_djg = djg1[(djg1['d_pm10'] <= Q3_djg['d_pm10']+1.5*IQR_djg['d_pm10']) & (djg1['d_pm10'] >= Q1_djg['d_pm10']-1.5*IQR_djg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

mpg= all_area[all_area['location']=='마포구']
mpg1 = pd.DataFrame(mpg)

Q1_mpg = mpg1[['d_pm10']].quantile(q=0.25)

Q3_mpg = mpg1[['d_pm10']].quantile(q=0.75)

IQR_mpg = Q3_mpg-Q1_mpg

IQR_df_mpg = mpg1[(mpg1['d_pm10'] <= Q3_mpg['d_pm10']+1.5*IQR_mpg['d_pm10']) & (mpg1['d_pm10'] >= Q1_mpg['d_pm10']-1.5*IQR_mpg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

sdmg= all_area[all_area['location']=='서대문구']
sdmg1 = pd.DataFrame(sdmg)

Q1_sdmg = sdmg1[['d_pm10']].quantile(q=0.25)

Q3_sdmg = sdmg1[['d_pm10']].quantile(q=0.75)

IQR_sdmg = Q3_sdmg-Q1_sdmg

IQR_df_sdmg = sdmg1[(sdmg1['d_pm10'] <= Q3_sdmg['d_pm10']+1.5*IQR_sdmg['d_pm10']) & (sdmg1['d_pm10'] >= Q1_sdmg['d_pm10']-1.5*IQR_sdmg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

scg= all_area[all_area['location']=='서초구']
scg1 = pd.DataFrame(scg)

Q1_scg = scg1[['d_pm10']].quantile(q=0.25)

Q3_scg = scg1[['d_pm10']].quantile(q=0.75)

IQR_scg = Q3_scg-Q1_scg

IQR_df_scg = scg1[(scg1['d_pm10'] <= Q3_scg['d_pm10']+1.5*IQR_scg['d_pm10']) & (scg1['d_pm10'] >= Q1_scg['d_pm10']-1.5*IQR_scg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

sdg= all_area[all_area['location']=='성동구']
sdg1 = pd.DataFrame(sdg)

Q1_sdg = sdg1[['d_pm10']].quantile(q=0.25)

Q3_sdg = sdg1[['d_pm10']].quantile(q=0.75)

IQR_sdg = Q3_sdg-Q1_sdg

IQR_df_sdg = sdg1[(sdg1['d_pm10'] <= Q3_sdg['d_pm10']+1.5*IQR_sdg['d_pm10']) & (sdg1['d_pm10'] >= Q1_sdg['d_pm10']-1.5*IQR_sdg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

sbg= all_area[all_area['location']=='성북구']
sbg1 = pd.DataFrame(sbg)

Q1_sbg = sbg1[['d_pm10']].quantile(q=0.25)

Q3_sbg = sbg1[['d_pm10']].quantile(q=0.75)

IQR_sbg = Q3_sbg-Q1_sbg

IQR_df_sbg = sbg1[(sbg1['d_pm10'] <= Q3_sbg['d_pm10']+1.5*IQR_sbg['d_pm10']) & (sbg1['d_pm10'] >= Q1_sbg['d_pm10']-1.5*IQR_sbg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

spg= all_area[all_area['location']=='송파구']
spg1 = pd.DataFrame(spg)

Q1_spg = spg1[['d_pm10']].quantile(q=0.25)

Q3_spg = spg1[['d_pm10']].quantile(q=0.75)

IQR_spg = Q3_spg-Q1_spg

IQR_df_spg = spg1[(spg1['d_pm10'] <= Q3_spg['d_pm10']+1.5*IQR_spg['d_pm10']) & (spg1['d_pm10'] >= Q1_spg['d_pm10']-1.5*IQR_spg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

ycg= all_area[all_area['location']=='양천구']
ycg1 = pd.DataFrame(ycg)

Q1_ycg = ycg1[['d_pm10']].quantile(q=0.25)

Q3_ycg = ycg1[['d_pm10']].quantile(q=0.75)

IQR_ycg = Q3_ycg-Q1_ycg

IQR_df_ycg = ycg1[(ycg1['d_pm10'] <= Q3_ycg['d_pm10']+1.5*IQR_ycg['d_pm10']) & (ycg1['d_pm10'] >= Q1_ycg['d_pm10']-1.5*IQR_ycg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

ydpg= all_area[all_area['location']=='영등포구']
ydpg1 = pd.DataFrame(ydpg)

Q1_ydpg = ydpg1[['d_pm10']].quantile(q=0.25)

Q3_ydpg = ydpg1[['d_pm10']].quantile(q=0.75)

IQR_ydpg = Q3_ydpg-Q1_ydpg

IQR_df_ydpg = ydpg1[(ydpg1['d_pm10'] <= Q3_ydpg['d_pm10']+1.5*IQR_ydpg['d_pm10']) & (ydpg1['d_pm10'] >= Q1_ydpg['d_pm10']-1.5*IQR_ydpg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

ysg= all_area[all_area['location']=='용산구']
ysg1 = pd.DataFrame(ysg)

Q1_ysg = ysg1[['d_pm10']].quantile(q=0.25)

Q3_ysg = ysg1[['d_pm10']].quantile(q=0.75)

IQR_ysg = Q3_ysg-Q1_ysg

IQR_df_ysg = ysg1[(ysg1['d_pm10'] <= Q3_ysg['d_pm10']+1.5*IQR_ysg['d_pm10']) & (ysg1['d_pm10'] >= Q1_ysg['d_pm10']-1.5*IQR_ysg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

epg= all_area[all_area['location']=='은평구']
epg1 = pd.DataFrame(epg)

Q1_epg = epg1[['d_pm10']].quantile(q=0.25)

Q3_epg = epg1[['d_pm10']].quantile(q=0.75)

IQR_epg = Q3_epg-Q1_epg

IQR_df_epg = epg1[(epg1['d_pm10'] <= Q3_epg['d_pm10']+1.5*IQR_epg['d_pm10']) & (epg1['d_pm10'] >= Q1_epg['d_pm10']-1.5*IQR_epg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

jrg= all_area[all_area['location']=='종로구']
jrg1 = pd.DataFrame(jrg)

Q1_jrg = jrg1[['d_pm10']].quantile(q=0.25)

Q3_jrg = jrg1[['d_pm10']].quantile(q=0.75)

IQR_jrg = Q3_jrg-Q1_jrg

IQR_df_jrg = jrg1[(jrg1['d_pm10'] <= Q3_jrg['d_pm10']+1.5*IQR_jrg['d_pm10']) & (jrg1['d_pm10'] >= Q1_jrg['d_pm10']-1.5*IQR_jrg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

jg= all_area[all_area['location']=='중구']
jg1 = pd.DataFrame(jg)

Q1_jg = jg1[['d_pm10']].quantile(q=0.25)

Q3_jg = jg1[['d_pm10']].quantile(q=0.75)

IQR_jg = Q3_jg-Q1_jg

IQR_df_jg = jg1[(jg1['d_pm10'] <= Q3_jg['d_pm10']+1.5*IQR_jg['d_pm10']) & (jg1['d_pm10'] >= Q1_jg['d_pm10']-1.5*IQR_jg['d_pm10'])]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

jlg= all_area[all_area['location']=='중랑구']
jlg1 = pd.DataFrame(jlg)

Q1_jlg = jlg1[['d_pm10']].quantile(q=0.25)

Q3_jlg = jlg1[['d_pm10']].quantile(q=0.75)

IQR_jlg = Q3_jlg-Q1_jlg

IQR_df_jlg = jlg1[(jlg1['d_pm10'] <= Q3_jlg['d_pm10']+1.5*IQR_jlg['d_pm10']) & (jlg1['d_pm10'] >= Q1_jlg['d_pm10']-1.5*IQR_jlg['d_pm10'])]

# ---------------------------------------------------------------------

data = [IQR_df_gng['d_pm10'], IQR_df_gdg['d_pm10'], IQR_df_gbg['d_pm10'], IQR_df_gsg['d_pm10'], IQR_df_gag['d_pm10'], IQR_df_gjg['d_pm10'], IQR_df_grg['d_pm10'], IQR_df_gcg['d_pm10'], IQR_df_nwg['d_pm10'], IQR_df_dbg['d_pm10'], IQR_df_ddmg['d_pm10'], IQR_df_djg['d_pm10'], IQR_df_mpg['d_pm10'], IQR_df_sdmg['d_pm10'], IQR_df_scg['d_pm10'], IQR_df_sdg['d_pm10'], IQR_df_sbg['d_pm10'], IQR_df_spg['d_pm10'], IQR_df_ycg['d_pm10'], IQR_df_ydpg['d_pm10'], IQR_df_ysg['d_pm10'], IQR_df_epg['d_pm10'], IQR_df_jrg['d_pm10'], IQR_df_jg['d_pm10'], IQR_df_jlg['d_pm10']]

seed = 42
X_train= to_time_series_dataset(data)
name_list = ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"]
# For this method to operate properly, prior scaling is required
# X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=365).fit_transform(X_train)
sz = X_train.shape[1]

dba_km = TimeSeriesKMeans(n_clusters=5,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)
print(y_pred)

result = [[], [], [], [], []]

for i, pre in enumerate(y_pred):
    if pre == 0:
        result[0].append(name_list[i])
    elif pre == 1:
        result[1].append(name_list[i])
    elif pre == 2:
        result[2].append(name_list[i])
    elif pre == 3:
        result[3].append(name_list[i])
    elif pre == 4:
        result[4].append(name_list[i])

print(result)

plt.figure(figsize=(14, 10))
for yi in range(5):
    plt.subplot(5, 1, 1+yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, sz, sz//8), ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"])
    plt.xlabel('연도(year)')
    plt.title('Cluster %d' % (yi+1))

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

ax.set_title("클러스터별 미세먼지 농도")
ax.boxplot([dba_km.cluster_centers_[0].ravel(), dba_km.cluster_centers_[1].ravel(), dba_km.cluster_centers_[2].ravel(), dba_km.cluster_centers_[3].ravel(), dba_km.cluster_centers_[4].ravel()])
ax.set_xlabel('클러스터')
ax.set_ylabel('미세먼지 농도(μg/m3)')

plt.show()

print(IQR_df_gng.columns)

datafr1 = pd.DataFrame({'date' : IQR_df_gng['date'],'d_pm10':IQR_df_gng['d_pm10'], 'location' : ["강남구"] * len(IQR_df_gng), 'cluster' : ["2"] * len(IQR_df_gng)})
datafr2 = pd.DataFrame({'date' : IQR_df_gdg['date'],'d_pm10':IQR_df_gdg['d_pm10'], 'location' : ["강동구"] * len(IQR_df_gdg), 'cluster' : ["2"] * len(IQR_df_gdg)})
datafr3 = pd.DataFrame({'date' : IQR_df_gbg['date'],'d_pm10':IQR_df_gbg['d_pm10'], 'location' : ["강북구"] * len(IQR_df_gbg), 'cluster' : ["1"] * len(IQR_df_gbg)})
datafr4 = pd.DataFrame({'date' : IQR_df_gsg['date'],'d_pm10':IQR_df_gsg['d_pm10'], 'location' : ["강서구"] * len(IQR_df_gsg), 'cluster' : ["2"] * len(IQR_df_gsg)})

datafr5 = pd.DataFrame({'date' : IQR_df_gag['date'],'d_pm10':IQR_df_gag['d_pm10'], 'location' : ["관악구"] * len(IQR_df_gag), 'cluster' : ["2"] * len(IQR_df_gag)})
datafr6 = pd.DataFrame({'date' : IQR_df_gjg['date'],'d_pm10':IQR_df_gjg['d_pm10'], 'location' : ["광진구"] * len(IQR_df_gjg), 'cluster' : ["1"] * len(IQR_df_gjg)})
datafr7 = pd.DataFrame({'date' : IQR_df_grg['date'],'d_pm10':IQR_df_grg['d_pm10'], 'location' : ["구로구"] * len(IQR_df_grg), 'cluster' : ["5"] * len(IQR_df_grg)})
datafr8 = pd.DataFrame({'date' : IQR_df_gcg['date'],'d_pm10':IQR_df_gcg['d_pm10'], 'location' : ["금천구"] * len(IQR_df_gcg), 'cluster' : ["1"] * len(IQR_df_gcg)})
datafr9 = pd.DataFrame({'date' : IQR_df_nwg['date'],'d_pm10':IQR_df_nwg['d_pm10'], 'location' : ["노원구"] * len(IQR_df_nwg), 'cluster' : ["2"] * len(IQR_df_nwg)})
datafr10 = pd.DataFrame({'date' : IQR_df_dbg['date'],'d_pm10':IQR_df_dbg['d_pm10'], 'location' : ["도봉구"] * len(IQR_df_dbg), 'cluster' : ["4"] * len(IQR_df_dbg)})
datafr11 = pd.DataFrame({'date' : IQR_df_ddmg['date'],'d_pm10':IQR_df_ddmg['d_pm10'], 'location' : ["동대문구"] * len(IQR_df_ddmg), 'cluster' : ["4"] * len(IQR_df_ddmg)})
datafr12 = pd.DataFrame({'date' : IQR_df_djg['date'],'d_pm10':IQR_df_djg['d_pm10'], 'location' : ["동작구"] * len(IQR_df_djg), 'cluster' : ["2"] * len(IQR_df_djg)})
datafr13 = pd.DataFrame({'date' : IQR_df_mpg['date'],'d_pm10':IQR_df_mpg['d_pm10'], 'location' : ["마포구"] * len(IQR_df_mpg), 'cluster' : ["2"] * len(IQR_df_mpg)})
datafr14 = pd.DataFrame({'date' : IQR_df_sdmg['date'],'d_pm10':IQR_df_sdmg['d_pm10'], 'location' : ["서대문구"] * len(IQR_df_sdmg), 'cluster' : ["2"] * len(IQR_df_sdmg)})
datafr15 = pd.DataFrame({'date' : IQR_df_scg['date'],'d_pm10':IQR_df_scg['d_pm10'], 'location' : ["서초구"] * len(IQR_df_scg), 'cluster' : ["1"] * len(IQR_df_scg)})
datafr16 = pd.DataFrame({'date' : IQR_df_sdg['date'],'d_pm10':IQR_df_sdg['d_pm10'], 'location' : ["성동구"] * len(IQR_df_sdg), 'cluster' : ["3"] * len(IQR_df_sdg)})
datafr17 = pd.DataFrame({'date' : IQR_df_sbg['date'],'d_pm10':IQR_df_sbg['d_pm10'], 'location' : ["성북구"] * len(IQR_df_sbg), 'cluster' : ["1"] * len(IQR_df_sbg)})
datafr18 = pd.DataFrame({'date' : IQR_df_spg['date'],'d_pm10':IQR_df_spg['d_pm10'], 'location' : ["송파구"] * len(IQR_df_spg), 'cluster' : ["3"] * len(IQR_df_spg)})
datafr19 = pd.DataFrame({'date' : IQR_df_ycg['date'],'d_pm10':IQR_df_ycg['d_pm10'], 'location' : ["양천구"] * len(IQR_df_ycg), 'cluster' : ["2"] * len(IQR_df_ycg)})
datafr20 = pd.DataFrame({'date' : IQR_df_ydpg['date'],'d_pm10':IQR_df_ydpg['d_pm10'], 'location' : ["영등포구"] * len(IQR_df_ydpg), 'cluster' : ["2"] * len(IQR_df_ydpg)})
datafr21 = pd.DataFrame({'date' : IQR_df_ysg['date'],'d_pm10':IQR_df_ysg['d_pm10'], 'location' : ["용산구"] * len(IQR_df_ysg), 'cluster' : ["2"] * len(IQR_df_ysg)})
datafr22 = pd.DataFrame({'date' : IQR_df_epg['date'],'d_pm10':IQR_df_epg['d_pm10'], 'location' : ["은평구"] * len(IQR_df_epg), 'cluster' : ["1"] * len(IQR_df_epg)})
datafr23 = pd.DataFrame({'date' : IQR_df_jrg['date'],'d_pm10':IQR_df_jrg['d_pm10'], 'location' : ["종로구"] * len(IQR_df_jrg), 'cluster' : ["2"] * len(IQR_df_jrg)})
datafr24 = pd.DataFrame({'date' : IQR_df_jg['date'],'d_pm10':IQR_df_jg['d_pm10'], 'location' : ["중구"] * len(IQR_df_jg), 'cluster' : ["2"] * len(IQR_df_jg)})
datafr25 = pd.DataFrame({'date' : IQR_df_jlg['date'],'d_pm10':IQR_df_jlg['d_pm10'], 'location' : ["중랑구"] * len(IQR_df_jlg), 'cluster' : ["2"] * len(IQR_df_jlg)})

datafr = pd.concat([datafr1, datafr2, datafr3, datafr4, datafr5, datafr6,datafr7, datafr8, datafr9,datafr10, datafr11, datafr12,datafr13, datafr14, datafr15,datafr16, datafr17, datafr18,datafr19, datafr20, datafr21,datafr22, datafr23, datafr24, datafr25])
datafr.to_csv('outlier_removal.csv')