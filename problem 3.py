# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:26:31 2022

@author: praja
"""

# Q3)
# import the libraries
import pandas as pd
import matplotlib.pylab as plt

# import the datasets
data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Data mining unsupervised learning hierarchical clustering\Telco_customer_churn.xlsx")

data.head(10)
data.info()       # it will give the information of all the null values present or not and the datatypes 

# perform the EDA

data.describe()

# remove the  CustomerID, count, quarter, refered a friend, no of referrals, paperless billing,payment method  as it is not useful for data analysis

data1 = data.drop(["Customer ID","Count","Quarter", "Referred a Friend","Number of Referrals","Paperless Billing", "Payment Method"], axis = 1)
data1.columns

# dividing the data into numeric and categorical data 

data_num = data1[['Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']]
data_num
data_cat = data1[['Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract',]]

# converting non numeric data into numeric

a = pd.get_dummies(data_cat, drop_first=True)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

data_norm = norm_func(data_num)
data_norm.describe()

# creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(18, 13));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
 
from sklearn.cluster import AgglomerativeClustering

data2 = AgglomerativeClustering(n_clusters = 3, linkage="complete", affinity ="euclidean").fit(data_norm)
data2.labels_                         # It will lables the clusters made

cluster_labels = pd.Series(data2.labels_)

# craeting new column and assigning it to the dataframe 

data_num['clust'] = cluster_labels  
 
# Rearranging the column name

data3 = data_num.iloc[:,[9,0,1,2,3,4,5,6,7,8]]

data3.head(10)

# aggregate mean of each cluster

mean_num = data.iloc[:, 1:].groupby(data3.clust).mean()

# creating csv file
data3.to_csv("Telecomunication churning rate1", encoding ="utf-8")

import os
os.getcwd()



# craeting clusters for non numeric data 

a = pd.get_dummies(data_cat, drop_first=True)


y = linkage(a, method = "complete", metric = "euclidean")

plt.figure(figsize=(18, 13));plt.title('Hierarchical Clustering Dendrogram for cat.data ');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(y, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
 
from sklearn.cluster import AgglomerativeClustering

b_cat = AgglomerativeClustering(n_clusters = 3, linkage="complete", affinity ="euclidean").fit(a)
b_cat.labels_                         # It will lables the clusters made

cluster_labels1 = pd.Series(b_cat.labels_)

# craeting new column and assigning it to the dataframe 

a['clust1'] = cluster_labels1  
 
# Rearranging the column name
b = a.iloc[:,[21,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

b.head(10)

# aggregate mean of each cluster
mean_cat = b.iloc[:,:].groupby(b.clust1).mean().transpose()
mean_cat
# creating csv file
b.to_csv("Telecomunication churning rate2", encoding ="utf-8")

import os
os.getcwd()











































################################################################






# convering the non numeric data to numeric data 
# rearranging the data 
data2 = data1.iloc[:,[1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] 
# label encodeng for offer columns as it is ordinal data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
a  = labelencoder.fit_transform(data1['Offer'])
y = pd.DataFrame(a)
y.rename(columns={ 0 : "Offer"},inplace= True)  ## rename the column name

data3 = pd.get_dummies(data2.iloc[:,1:], drop_first = True)

# concate data2 and 3 
 data4 = pd.concat([y,data3],axis=1)

# normalisation of data 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
data_norm = norm_func(data4)
data_norm.describe()

# creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(18, 13));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
 
from sklearn.cluster import AgglomerativeClustering

data5 = AgglomerativeClustering(n_clusters = 3, linkage="complete", affinity ="euclidean").fit(data_norm)
data5.labels_                         # It will lables the clusters made

cluster_labels = pd.Series(data5.labels_)

# craeting new column and assigning it to the dataframe 

data1['clust'] = cluster_labels  
 
# Rearranging the column name

data = data1.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]

data.head(10)

# aggregate mean of each cluster
 a = data.iloc[:, 1:].groupby(data.clust).mean().transpose()
a
# creating csv file
data.to_csv("Telecomunication churning rate", encoding ="utf-8")

import os
os.getcwd()
help(data.groupby)

data1 = data.drop(["Customer ID","Count","Quarter", "Referred a Friend","Number of Referrals","Paperless Billing", "Payment Method"], axis = 1)
data1.columns

data_num = data1[['Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']]
data_cat = data1[['Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract',]]


r = pd.get_dummies(data_cat, drop_first = True)

# creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(r, method = "complete", metric = "euclidean")

plt.figure(figsize=(18, 13));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
 
from sklearn.cluster import AgglomerativeClustering

data5= AgglomerativeClustering(n_clusters = 3, linkage="complete", affinity ="euclidean").fit(r)
data5.labels_                         # It will lables the clusters made

cluster_labels = pd.Series(data5.labels_)

# craeting new column and assigning it to the dataframe 

r['clust'] = cluster_labels  
 
# Rearranging the column name

data = data1.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]

data.head(10)

# aggregate mean of each cluster
 a = r.iloc[:, 0:21].groupby(r.clust).mean().transpose()
a

 a = r.iloc[:, 0:21].groupby(r.clust).mode().transpose()
