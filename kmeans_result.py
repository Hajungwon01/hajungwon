import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler


plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame()
df = pd.read_csv('exclude_null.csv')

df=df.drop([df.columns[0]], axis=1)
df['date']=pd.to_datetime(df['date'], infer_datetime_format=True)

df = df.reset_index(drop = True)

seoul=df.copy()

gng=seoul[seoul['location']=='강남구']
gng1=gng['d_pm10']

gdg=seoul[seoul['location']=='강동구']
gdg1=gdg['d_pm10']

gbg= seoul[seoul['location']=='강북구']
gbg1=gbg['d_pm10']

gsg= seoul[seoul['location']=='강서구']
gsg1=gsg['d_pm10']

gag= seoul[seoul['location']=='관악구']
gag1=gag['d_pm10']

gjg= seoul[seoul['location']=='광진구']
gjg1=gjg['d_pm10']

grg= seoul[seoul['location']=='구로구']
grg1=grg['d_pm10']

gcg= seoul[seoul['location']=='금천구']
gcg1=gcg['d_pm10']

nwg= seoul[seoul['location']=='노원구']
nwg1=nwg['d_pm10']

dbg= seoul[seoul['location']=='도봉구']
dbg1=dbg['d_pm10']

ddmg= seoul[seoul['location']=='동대문구']
ddmg1=ddmg['d_pm10']

djg= seoul[seoul['location']=='동작구']
djg1=djg['d_pm10']

mpg= seoul[seoul['location']=='마포구']
mpg1=mpg['d_pm10']

sdmg= seoul[seoul['location']=='서대문구']
sdmg1=sdmg['d_pm10']

scg= seoul[seoul['location']=='서초구']
scg1=scg['d_pm10']

sdg= seoul[seoul['location']=='성동구']
sdg1=sdg['d_pm10']

sbg= seoul[seoul['location']=='성북구']
sbg1=sbg['d_pm10']

spg= seoul[seoul['location']=='송파구']
spg1=spg['d_pm10']

ycg= seoul[seoul['location']=='양천구']
ycg1=ycg['d_pm10']

ydpg= seoul[seoul['location']=='영등포구']
ydpg1=ydpg['d_pm10']

ysg= seoul[seoul['location']=='용산구']
ysg1=ysg['d_pm10']

epg= seoul[seoul['location']=='은평구']
epg1=epg['d_pm10']

jrg= seoul[seoul['location']=='종로구']
jrg1=jrg['d_pm10']

jg= seoul[seoul['location']=='중구']
jg1=jg['d_pm10']

jlg= seoul[seoul['location']=='중랑구']
jlg1=jlg['d_pm10']

seed = 0
numpy.random.seed(seed)
X_train= to_time_series_dataset([gng1, gdg1, gbg1, gsg1, gag1, gjg1, grg1, gcg1, nwg1, dbg1, ddmg1, djg1, mpg1, sdmg1, scg1, sdg1, sbg1, spg1, ycg1, ydpg1, ysg1, epg1, jrg1, jg1, jlg1])
name_list = ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"]
numpy.random.shuffle(X_train)
# For this method to operate properly, prior scaling is required
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=20).fit_transform(X_train)
sz = X_train.shape[1]

dba_km = TimeSeriesKMeans(n_clusters=5,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)
print(y_pred)

date = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2023"]

for yi in range(5):
    plt.subplot(5, 1, 1+yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.title('Cluster %d' % (yi+1))
    

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

plt.tight_layout()
plt.show()