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
df = pd.read_csv('outlier_removal.csv')

 
df.info()   

df = df.reset_index(drop = True)

all_area=df.copy()

cluster1 = all_area[(all_area['location']=='강북구') | (all_area['location']=='광진구') | (all_area['location']=='금천구') | (all_area['location']=='서초구') | (all_area['location']=='성북구') | (all_area['location']=='은평구')]['d_pm10']
cluster2 = all_area[(all_area['location']=='강북구') | (all_area['location']=='강동구') | (all_area['location']=='강서구') | (all_area['location']=='관악구') | (all_area['location']=='노원구') | (all_area['location']=='동작구') | (all_area['location']=='마포구') | (all_area['location']=='서대문구') | (all_area['location']=='양천구') | (all_area['location']=='영등포구') | (all_area['location']=='용산구') | (all_area['location']=='종로구') | (all_area['location']=='중구') | (all_area['location']=='중랑구')]['d_pm10']
cluster3 = all_area[(all_area['location']=='성동구') | (all_area['location']=='송파구')]['d_pm10']
cluster4 = all_area[(all_area['location']=='도봉구') | (all_area['location']=='동대문구')]['d_pm10']
cluster5 = all_area[(all_area['location']=='구로구')]['d_pm10']

# -------------------------------------------------------------------------------------------------

cluster1 = np.array(cluster1).flatten()
cluster2 = np.array(cluster2).flatten()
cluster3 = np.array(cluster3).flatten()
cluster4 = np.array(cluster4).flatten()
cluster5 = np.array(cluster5).flatten()

print('cluster 0의 평균 : ' ,  np.mean(cluster1))
print('cluster 1의 평균 : ' , np.mean(cluster2))
print('cluster 2의 평균 : ' ,np.mean(cluster3))
print('cluster 3의 평균 : ' ,np.mean(cluster4))
print('cluster 4의 평균 : ' ,np.mean(cluster5))

print('cluster 0의 표준편차 : ' ,np.std(cluster1))
print('cluster 1의 표준편차 : ' ,np.std(cluster2))
print('cluster 2의 표준편차 : ' ,np.std(cluster3))
print('cluster 3의 표준편차 : ' ,np.std(cluster4))
print('cluster 4의 표준편차 : ' ,np.std(cluster5))

stat = pd.DataFrame({'cluster 1' : [np.mean(cluster1), np.std(cluster1)], 'cluster 2' : [np.mean(cluster2), np.std(cluster2)], 'cluster 3' : [np.mean(cluster3), np.std(cluster3)], 'cluster 4' : [np.mean(cluster4), np.std(cluster4)], 'cluster 5' : [np.mean(cluster5), np.std(cluster5)]}, index = ["평균", "표준편차"])
print(stat)