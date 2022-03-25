# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:01:07 2022

@author: praja
"""
Q1) # import the libraries
import pandas as pd
import matplotlib.pylab as plt

# import the data 
# Q1) importing the libraries
import pandas as pd
import matplotlib.pylab as plt

# import the datasets
data = pd.read_excel(r"C:\Users\praja\Desktop\EastWestAirlines.xlsx")

data.head(10)
data.info()       # it will give the information of all the null values present or not and the datatypes 

# perform the EDA

data.describe().transpose()

# remove the ID and award as it is not useful for data analysis

data1 = data.drop(['ID#','Award?'], axis=1)
# normalisation of data 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
data_norm = norm_func(data1)
data_norm.describe()

# creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(25, 6));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
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

data1['clust'] = cluster_labels  
 
# Rearranging the column name

data = data1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]

data.head(20)

# aggregate mean of each cluster
 a = data.iloc[:, 1:].groupby(data.clust).mean().transpose()

# creating csv file
data.to_csv("Airline_eastwest", encoding ="utf-8")

import os
os.getcwd()
####################################################################

Q2) 
    # import the libraries
import pandas as pd
import matplotlib.pylab as plt


# import the datasets
data = pd.read_csv(r"C:/Users/praja/Desktop/Data Science/Data mining unsupervised learning hierarchical clustering/crime_data.csv")

data.head(10)
data.info()       # it will give the information of all the null values present or not and the datatypes 

# perform the EDA

data.describe().transpose()

# remove the ID and award as it is not useful for data analysis


# normalisation of data 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
data_norm = norm_func(data.iloc[:,1:])
data_norm.describe().transpose()

# creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
 
from sklearn.cluster import AgglomerativeClustering

data1 = AgglomerativeClustering(n_clusters = 4, linkage="complete", affinity ="euclidean").fit(data_norm)
data1.labels_                         # It will lables the clusters made

cluster_labels = pd.Series(data1.labels_)

# craeting new column and assigning it to the dataframe 

data['clust'] = cluster_labels  
 
# Rearranging the column name

data.iloc[:,[5,0,1,2,3,4]]

data.head(10)

# aggregate mean of each cluster
data.iloc[:, 1:].groupby(data.clust).mean().transpose()

# creating csv file
data.to_csv("crime_rate", encoding ="utf-8")

import os
os.getcwd()
 ##################################################################


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

mean_num = data.iloc[:, :].groupby(data3.clust).mean()
mean_num.transpose()


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

data3.to_csv("Telecomunication churning rate1", encoding ="utf-8")

import os
os.getcwd()

#################################################################################

Q) 4
  # import the libraries
import pandas as pd
import matplotlib.pylab as plt

# import the datasets
data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data mining unsupervised learning hierarchical clustering\AutoInsurance.csv")
data.head(10)
data.info()       # it will give the information of all the null values present or not and the datatypes 

# perform the EDA

data.describe().transpose()
data.columns
# remove the  'Customer state, lobcation code, quarter, refered a friend, no of referrals, paperless billing,payment method  as it is not useful for data analysis

data1 = data.drop(['Customer','State', 'Marital Status'], axis = 1)

data1.columns
# convert date function to no of days 
import datetime
data1['Effective To Date']= pd.to_datetime(data1["Effective To Date"])
data1.info()

import datetime

today = datetime.datetime.today()
today

data1['Effective To Date'] = (today- data1["Effective To Date"]).dt.days

# dividing the data into numeric and categorical data 
data1.columns

data_num = data1[['Customer Lifetime Value','Effective To Date','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints', 'Number of Policies','Total Claim Amount']]
data_num
data_cat = data1[['Response','Coverage','Education','EmploymentStatus','Gender','Location Code','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class', 'Vehicle Size']]

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

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram data_num');plt.xlabel('Index');plt.ylabel('Distance')
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

mean_num = data.iloc[:, :].groupby(data3.clust).mean()
mean_num.transpose()


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
b = a.iloc[:,[37,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]]

b.head(10)

# aggregate mean of each cluster
mean_cat = b.iloc[:,:].groupby(b.clust1).mean().transpose()
mean_cat
# creating csv file
b.to_csv("Autoinsurance2", encoding ="utf-8")

data3.to_csv("Autoinsurance1", encoding ="utf-8")

import os
os.getcwd()
  
  