#!/usr/bin/env python
# coding: utf-8

# @author Fatma Erkan

# In[2]:


#Libraries 
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


filename = "20191218-30min-PassengerCount.csv"
df = pd.read_csv(filename)


# ## Data Analysis

# In[4]:


df


# In[5]:


pd.unique(df["BoardingTime"])


# In[6]:


def time_conv(d):
    return (datetime.strptime(d, '%Y-%m-%dT%H:%M'))


# In[7]:


df['BoardingTime'] = df['BoardingTime'].apply(time_conv)


# In[8]:


dfG = df.groupby("Line")
keys = list(dfG.groups.keys())
fig, ax = plt.subplots()
for key, grp in dfG:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=4,prop={'size': 9})


# In this graph each line shows different routes. Here, we can see that some lines are visibly have higher values than other lines. There is also some noticable trends, for example, there are noticable spikes between 6-9 am and 3-6 pm for all lines, which are rush hours. 

# In[9]:


dfmin = df.sort_values('PassengerCount', ascending=False).drop_duplicates(['BoardingTime'])
dfmin.sort_index(inplace=True)
print("Bus lines with maximum passengers for first 6 time points:")
print("----------------------------------------------------------")
print(dfmin.head())
print("----------------------------------------------------------")
print("Bus lines with maximum occurances in the total version of above list:")
print(dfmin['Line'].value_counts())


# In[10]:


toP = dfmin['Line'].value_counts()[:5].reset_index()['index']
fig, ax = plt.subplots()
ax = df.groupby("BoardingTime").mean().plot(ax=ax, kind='line', label="average",linewidth=5.0)
for key in toP:
    dftp = dfG.get_group(key)
    ax = dftp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key)
L=plt.legend()
L.get_texts()[0].set_text('Average')


# In[11]:


dfmax = df.sort_values('PassengerCount', ascending=True).drop_duplicates(['BoardingTime'])
dfmax.sort_index(inplace=True)
print("Bus lines with minimum recorded passengers for first 6 time points:")
print(dfmax.head())
print("-------------------------------------------------------------------")
print("Bus lines with maximum occurances in the total version of above list:")
print(dfmax['Line'].value_counts()[:8])


# In[12]:


toP = dfmax['Line'].value_counts()[:5].reset_index()['index']
fig, ax = plt.subplots()
ax = df.groupby("BoardingTime").mean().plot(ax=ax, kind='line',linewidth=5.0)
for key in toP:
    dftp = dfG.get_group(key)
    ax = dftp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key)
L=plt.legend()
L.get_texts()[0].set_text('Average')


# In[13]:


dfs = dfG.sum().sort_values("PassengerCount",ascending=False).reset_index()
dfs.index += 1 
print("Lines with most total passengers:")
print(dfs.head())
print("---------------------------------")
print("Lines with least total passengers:")
print(dfs.tail().iloc[::-1])


# From the last 3 results, we can say that the most busy line is KLo8 line. For the least busy, there are multiple lines with similarly low amount of passengers, and among them KC35 and TC93 are noticable.

# In[14]:


for name, group in dfG:
    plt.plot(group["BoardingTime"], group["PassengerCount"], marker="o", linestyle="", label=name)


# In[15]:


#In this scatter plot each line's data is shown in the same color. This plot can give us insight about the total scatter plot in which clustering will be done.


# ## Correlation Analysis of Bus Lines Passenger Data

# In[16]:


start=1
for name,grp in dfG:
    temp = grp.copy().set_index("BoardingTime").drop(columns="Line").rename(columns={'PassengerCount': name})
    if start==1:
        dfTemp = temp
        start = 0
    else:
        dfTemp[name] = temp
dfTemp = dfTemp.fillna(0)
dfcorr = dfTemp.corr().values
print("Average of all correlation coefficients between bus lines:")
print(dfcorr.mean())
print("----------------------------------------------------------")
print("Standart deviation of all correlation coefficients between bus lines:")
print(dfcorr.std())


# From the correalation analysis, we can say that all of the bus lines have very similar characteristic. This result is in line with previous results and implies that for rasping the general pattern of the date, focusing on time would be heplful.

# # k-means Clustering

# In[17]:


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# In[18]:


def time_diff(a,date_min):
    res = (a-date_min).seconds/60 + (a-date_min).days*1440
    return(res)


# Here, time is taken in units of minutes. This decision is important, as the unit of time affects distance calculations within k-means algorithm.

# In[19]:


dfc = df.copy()
date_min = dfc.BoardingTime.min()
dfc['BoardingTime'] = df['BoardingTime'].apply(time_diff,date_min=date_min)
le = LabelEncoder()
dfc['Line'] = le.fit_transform(dfc['Line'])
#le.inverse_transform(dfc['Line'])
clusters = KMeans(n_clusters=6, random_state=0).fit_predict(dfc)
df2 = df.copy()
df2['cluster'] = clusters
dfG2 = df2.groupby("cluster")
fig, ax = plt.subplots()
for key, grp in dfG2:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")

ax.get_legend().remove()


# Here different clusters are shown in different colors. the number of clusters is choosen after trying a few numbers, and 6 was choosen as it shows the affect of time clearly. Some clusters (top two) are mainly differentiated from others based on their amount of passengers, but the rest (and the top two between themselves) are seperated are mainly seperated by time

# ### Changing Weights 

# In[20]:


#seconds
def time_diff2(a,date_min):
    res = (a-date_min).seconds + (a-date_min).days*86400
    return(res)
#hours
def time_diff3(a,date_min):
    res = (a-date_min).seconds/3600 + (a-date_min).days*24
    return(res)


# In[21]:


dfc2 = df.copy()
date_min = dfc2.BoardingTime.min()
dfc2['BoardingTime'] = df['BoardingTime'].apply(time_diff2,date_min=date_min)
le = LabelEncoder()
dfc2['Line'] = le.fit_transform(dfc['Line'])
clusters = KMeans(n_clusters=6, random_state=0).fit_predict(dfc2)
dff = df.copy()
dff['cluster'] = clusters
dfGf = dff.groupby("cluster")
fig, ax = plt.subplots()
for key, grp in dfGf:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")
#plt.legend(ncol=5)
ax.get_legend().remove()


# In[22]:


dfc2 = df.copy()
date_min = dfc2.BoardingTime.min()
dfc2['BoardingTime'] = df['BoardingTime'].apply(time_diff3,date_min=date_min)
le = LabelEncoder()
dfc2['Line'] = le.fit_transform(dfc['Line'])
clusters = KMeans(n_clusters=6, random_state=0).fit_predict(dfc2)
dff = df.copy()
dff['cluster'] = clusters
dfGf = dff.groupby("cluster")
fig, ax = plt.subplots()
for key, grp in dfGf:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")
#plt.legend(ncol=5)
ax.get_legend().remove()


# Here, other time units are investigated. Firstly, the unit for time is selected as seconds, which created clusters entirely on time, with equal time difference for each cluster (nearly 3 hours). The resulting line graph is also included. Then, a time unit of hour is used. This time the clusters are observed entirely horizontally (only based on number of passenger data). Because of these two clusterings are simplifying the difference to only one feature, it can be argued that selecting minutes as the time unit is sensible. However, further investigation on time units that are multiples of 1 minute are not investigated as the obtained results seemed sufficient.

# # Partitinal Clustering Method - Mean Shift Clustering

# In[23]:


from sklearn.cluster import MeanShift


# In[24]:


df2['cluster'] = MeanShift().fit_predict(dfc)
dfG3 = df2.groupby("cluster")
fig, ax = plt.subplots()
for key, grp in dfG3:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")
ax.get_legend().remove()


# Here, another partitional clustering method is used. In k-means method, there is a risk of dividing a cluster to multiple clusters and in this method, mean shift, this risk does not exist. The result obtained from this method shows the affect of time even more clearly.

# # Agglomerative Hierarchical Clustering

# In[25]:


from sklearn.cluster import AgglomerativeClustering


# In[26]:


df2['c1'] = AgglomerativeClustering(n_clusters=6,linkage="ward").fit_predict(dfc)
dfG3 = df2.groupby("c1")
fig, ax = plt.subplots()
for key, grp in dfG3:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")
ax.get_legend().remove()


# In[27]:


df2['c1'] = AgglomerativeClustering(n_clusters=7,linkage="ward").fit_predict(dfc)
dfG3 = df2.groupby("c1")
fig, ax = plt.subplots()
for key, grp in dfG3:
    ax = grp.plot(ax=ax, kind='line', x='BoardingTime', y='PassengerCount', label=key,marker="o", linestyle="")
ax.get_legend().remove()


# Hierarchical methods are a different kind of clustering method, which uses an iterative method to either combine smaller clusters or divide a cluster to a given number. Because of their method, they are not especially inclined to spherical clusters (such as circles and spheres) which can be seen as an advantage compared to k-means method. Here we can see sharper dividers between clusters and we can also see a new cluster at the smaller values for the earlier hours. The other divider between top groups can again be seen if the number of clusters is changed to 7.
 



