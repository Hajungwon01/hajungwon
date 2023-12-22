#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:29:24 2023

@author: jhkim94
"""

# from tslearn.datasets import UCR_UEA_datasets
# data_loader = UCR_UEA_datasets()
# X_train, y_train, X_test, y_test = data_loader.load_dataset("Adiac")
# from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
# dataset = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3]])
# save_time_series_txt("tmp-tslearn-test.txt", dataset)

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
from tslearn.utils import save_time_series_txt, load_time_series_txt, to_time_series_dataset
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import collections
from matplotlib import pyplot
from datetime import date
import holidays
import prophet
from prophet.plot import plot_yearly
from prophet.plot import plot_seasonality
from datetime import datetime
from prophet.plot import add_changepoints_to_plot
import matplotlib.font_manager 
import numpy as np
# plt.rcParams['font.family']='NanumGothic'

df = pd.DataFrame()
df = pd.read_csv('exclude_null.csv')

df=df.drop([df.columns[0]], axis=1)
df['date']=pd.to_datetime(df['date'], infer_datetime_format=True)
 
df.info()   

    
df = df.reset_index(drop = True)

loca=df['location'].unique()
print(loca)
seoul=df.copy()

temp=pd.DataFrame()

""" for loc in loca:
    plt.figure(figsize=(20, 10))
    temp=seoul[seoul['location']==loc]
    plt.plot(temp['date'], temp['d_pm10'])
    plt.title(loc)
    plt.show() """
    
# km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0).fit(X)
# km.cluster_centers_.shape
# km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=0).fit(X)
# km_dba.cluster_centers_.shape
# km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,
#                            max_iter_barycenter=5,
#                            metric_params={"gamma": .5},
#                            random_state=0).fit(X)
# km_sdtw.cluster_centers_.shape

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


n_cluster = 5
XY= to_time_series_dataset([gng1, gdg1, gbg1, gsg1, gag1, gjg1, grg1, gcg1, nwg1, dbg1, ddmg1, djg1, mpg1, sdmg1, scg1, sdg1, sbg1, spg1, ycg1, ydpg1, ysg1, epg1, jrg1, jg1, jlg1])
km = TimeSeriesKMeans(n_clusters=n_cluster, max_iter=5, metric="dtw", random_state=0, n_jobs=-1).fit(XY)
y_pred = km.predict(XY)

print(y_pred)

labels = []
sizes = []
for i in range(n_cluster):
  labels.append("cluster_"+str(i))
  sizes.append(collections.Counter(y_pred)[i])

plt.figure(figsize=(10,5))
plt.pie(sizes,labels=labels,shadow=False,startangle=90,autopct='%1.1f%%')
plt.title("test",position=(0.5,1.2),fontsize=20)
plt.show()


km.cluster_centers_.shape
print(km.cluster_centers_)