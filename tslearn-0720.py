#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:07:08 2023

@author: jhkim94
"""


from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
import pandas as pd
import matplotlib.pyplot as plt
# from prophet import Prophet
import os
import collections
from matplotlib import pyplot
from datetime import date
# import holidays
# import prophet
# from prophet.plot import plot_yearly
# from prophet.plot import plot_seasonality
from datetime import datetime
# from prophet.plot import add_changepoints_to_plot
import matplotlib.font_manager 
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
plt.rcParams['font.family']='NanumGothic'

df = pd.DataFrame()
df = pd.read_csv('exclude_null.csv')

df['date']=pd.to_datetime(df['date'], infer_datetime_format=True)
 
df.info()   

df = df.reset_index(drop = True)

loca=df['location'].unique()
# print(loca)
all_area=df.copy()

temp=pd.DataFrame()

""" for loc in loca:
    plt.figure(figsize=(20, 10))
    temp=all_area[all_area['location']==loc]
    plt.plot(temp['date'], temp['d_pm10'])
    plt.title(loc)
    plt.show()  """
    
# km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0).fit(X)
# km.cluster_centers_.shape
# km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=0).fit(X)
# km_dba.cluster_centers_.shape
# km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,
#                            max_iter_barycenter=5,
#                            metric_params={"gamma": .5},
#                            random_state=0).fit(X)
# km_sdtw.cluster_centers_.shape

gng=all_area[all_area['location']=='강남구']
gng1=gng['d_pm10']

gdg=all_area[all_area['location']=='강동구']
gdg1=gdg['d_pm10']

gbg= all_area[all_area['location']=='강북구']
gbg1=gbg['d_pm10']

gsg= all_area[all_area['location']=='강서구']
gsg1=gsg['d_pm10']

gag= all_area[all_area['location']=='관악구']
gag1=gag['d_pm10']

gjg= all_area[all_area['location']=='광진구']
gjg1=gjg['d_pm10']

grg= all_area[all_area['location']=='구로구']
grg1=grg['d_pm10']

gcg= all_area[all_area['location']=='금천구']
gcg1=gcg['d_pm10']

nwg= all_area[all_area['location']=='노원구']
nwg1=nwg['d_pm10']

dbg= all_area[all_area['location']=='도봉구']
dbg1=dbg['d_pm10']

ddmg= all_area[all_area['location']=='동대문구']
ddmg1=ddmg['d_pm10']

djg= all_area[all_area['location']=='동작구']
djg1=djg['d_pm10']

mpg= all_area[all_area['location']=='마포구']
mpg1=mpg['d_pm10']

sdmg= all_area[all_area['location']=='서대문구']
sdmg1=sdmg['d_pm10']

scg= all_area[all_area['location']=='서초구']
scg1=scg['d_pm10']

sdg= all_area[all_area['location']=='성동구']
sdg1=sdg['d_pm10']

sbg= all_area[all_area['location']=='성북구']
sbg1=sbg['d_pm10']

spg= all_area[all_area['location']=='송파구']
spg1=spg['d_pm10']

ycg= all_area[all_area['location']=='양천구']
ycg1=ycg['d_pm10']

ydpg= all_area[all_area['location']=='영등포구']
ydpg1=ydpg['d_pm10']

ysg= all_area[all_area['location']=='용산구']
ysg1=ysg['d_pm10']

epg= all_area[all_area['location']=='은평구']
epg1=epg['d_pm10']

jrg= all_area[all_area['location']=='종로구']
jrg1=jrg['d_pm10']

jg= all_area[all_area['location']=='중구']
jg1=jg['d_pm10']

jlg= all_area[all_area['location']=='중랑구']
jlg1=jlg['d_pm10']

"""

mhd= all_area[all_area['location']=='모현동']
mhd1=mhd['d_pm10']

pbd= all_area[all_area['location']=='팔봉동']
pbd1=pbd['d_pm10']

yd= all_area[all_area['location']=='연동']
yd1=yd['d_pm10']

ldd= all_area[all_area['location']=='이도동']
ldd1=ldd['d_pm10']


dhd= all_area[all_area['location']=='동홍동']
dhd1=dhd['d_pm10']
dhd2=dhd['d_pm25']

idd= all_area[all_area['location']=='모현동']
idd1=idd['d_pm10']
idd2=idd['d_pm25']
"""

n_cluster = 5
XY= to_time_series_dataset([gng1, gdg1, gbg1, gsg1, gag1, gjg1, grg1, gcg1, nwg1, dbg1, ddmg1, djg1, mpg1, sdmg1, scg1, sdg1, sbg1, spg1, ycg1, ydpg1, ysg1, epg1, jrg1, jg1, jlg1])
np.random.shuffle(XY)
# X_train=TimeSeriesScalerMeanVariance().fit_transform(XY)


km = TimeSeriesKMeans(n_clusters=n_cluster, max_iter=150, metric="dtw", random_state=0, verbose=True, n_jobs=-1).fit(XY)
y_pred = km.predict(XY)

labels = []
sizes = []
for i in range(n_cluster):
  labels.append("cluster_"+str(i))
  sizes.append(collections.Counter(y_pred)[i])

print(sizes)

plt.figure(figsize=(10,5))
plt.pie(sizes,labels=labels,shadow=False,startangle=90,autopct='%1.1f%%')
plt.title("test",position=(0.5,1.2),fontsize=20)
plt.show()

def show_cluster_dist(num_cluster):
  plt.figure(figsize=(10,4))
  size = sizes[num_cluster]
  for i in range(size):
    name = "cluster" + str(i) 
    plt.plot(XY[y_pred==num_cluster][i])
  plt.ylabel("미세먼지의 양")
  plt.legend()
  plt.show()


print(y_pred)