# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:00:32 2019

@author: Ghous
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values


#Use elbow method to find optimal number of clusters

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    #interia function returns the wcss value of our kmean classification result. 
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.ylabel('WCSS value')
plt.xlabel('Number of cluster')
plt.show()
    

#Applying K mean to our dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
#fit_predict returns the cluster number for every data point. 
y_kmeans = kmeans.fit_predict(x)


#visualising the clusters.
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1' )
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2' )
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3' )
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4' )
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5' )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='cluster_centre')
plt.title('Cluster of clients')
plt.xlabel('Annual income ($k)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()