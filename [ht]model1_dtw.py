#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from statsmodels.tsa.seasonal import seasonal_decompose
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw_path_from_metric
import json


# In[95]:


# Read the data, pass as X
df = pd.read_csv('hanbell.csv', sep=",")
X = df[['Time','Exhaust pressure']]


# In[96]:


time = np.array(X.T)[0]
data = np.array(X.T)[1]
# datat = np.array(X.T)[1] # for plotting purpose, can be eliminated in the final model
timestamp_for_plot = pd.to_datetime(time)


# In[97]:


# finding the frequency / daily case
timestamp_in_date = pd.to_datetime(time).date
counting_the_date = np.unique(timestamp_in_date, return_counts=True)
freq = np.max(counting_the_date[1])


# In[98]:


# data compression
n = 10
w = np.int(freq/n)

paa = PiecewiseAggregateApproximation(window_size=n)
data_compressed = paa.transform(data.reshape(1,-1))[0]


# In[99]:


true_sub_id = np.array([0])
for i in counting_the_date[1]:
    true_sub_id = np.append(true_sub_id, np.max(true_sub_id)+i)
    
sub_id = (true_sub_id/n).astype(int)


# In[100]:


# trend removal
res = seasonal_decompose(data_compressed, model='additive', period=w, extrapolate_trend='freq')
data_cmpsd_trm = data_compressed/res.trend


# In[101]:


# determining the subsection
adj_sub_id = np.ceil(sub_id).astype(int)

# create the list of elements of the subseq
subseq = []
for i in range(1, len(adj_sub_id)):
    #print(adj_sub_id[i-1], adj_sub_id[i])
    subseq.append(data_cmpsd_trm[(adj_sub_id[i-1]):(adj_sub_id[i])])


# In[102]:


# finding the maximum len of subsequences
maxLen = 0
for j in range(len(subseq)):
    if len(subseq[j]) > maxLen:
        maxLen = len(subseq[j])
        
data = np.empty((len(subseq), maxLen, 1))
for i in range(len(subseq)):
    if len(subseq[i]) < maxLen :
        elements = np.array( list(subseq[i]) + [np.nan]*((maxLen)-len(subseq[i])) )
    else :
        elements = np.array( list(subseq[i]) )
    data[i].T[0] = elements


# In[103]:


# calculate the pairwise distance between subsequences based on dtw
distances = np.array([])
for i in range(len(data)):
    d = 0
    for j in range(len(data)):
        d += dtw_path_from_metric(data[i], data[j])[1]
    distances = np.append(distances, d)

distances[0] = distances[-1] = np.median(distances) # avoiding reporting the first and last observations, potential of non-completed data


# In[104]:


# contruct the UCL limit 
UCL = np.mean(distances) + 3*np.std(distances)/np.sqrt(len(distances))


# In[105]:


y = np.array([0]*len(distances))
y[distances>UCL] = 1


# In[106]:


# ft1, ft2 = 25, 20

# x = np.arange(len(datat))+1
# plt.figure(figsize=(20,3))

# plt.plot(timestamp_for_plot, datat, c='k')
# plt.ylabel("Value", fontsize=ft1)
# plt.tick_params(axis='x', which='major', labelsize=0)
# plt.tick_params(axis='y', which='major', labelsize=ft2)
# for i in range(len(y)):
#     if y[i] == 1 :
#         plt.plot(timestamp_for_plot[true_sub_id[i-1]:true_sub_id[i]], 
#                  datat[true_sub_id[i-1]:true_sub_id[i]], c='r')

# plt.xlabel('Time', fontsize=ft1)
# plt.tick_params(axis='x', which='major', labelsize=ft2)
# # plt.savefig('example_output_for_viewing_the_discord.png', bbox_inches='tight', dpi=600)


# In[111]:


#print("--- %s seconds ---" % (tm.time() - start_time))


# In[108]:


# return >> dataset and discords
dataset = {}
dataset['index'] = list(range(len(X)))
dataset['timestamp'] = list(np.array(X.T)[0])
dataset['value'] = list(np.array(X.T)[1])

# reporting the discord

location = []
#d_start = []
#d_end = []
if np.sum(y) > 0 :
    ls = np.where(y>0)
    for i in ls[0]:
        #print(sub_id[i-1], sub_id[i])
        location.append([sub_id[i-1], sub_id[i]])
        #d_start.append(sub_id[i-1])
        #d_end.append(sub_id[i])
else :
    pass

discords = {}
#kIdx = []
for i in range(len(location)):
    #kIdx.append(i)
    discords[i] = location[i]
print(discords)


# In[109]:


# d_discord = [kIdx,d_start, d_end]
# print(d_discord)


# In[ ]:





# In[ ]:





# In[ ]:




