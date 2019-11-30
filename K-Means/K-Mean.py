import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
from sklearn.cluster import KMeans

data=pd.read_csv("Countries-exercise.csv")
# print(data)
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
# plt.show()

x=data.iloc[:,1:3]
# print(x)

kmeans=KMeans(2)
kmeans.fit(x)

identified_clusters=kmeans.fit_predict(x)
# print(identified_clusters)

data_with_clusters=data.copy()
data_with_clusters['Cluster']=identified_clusters
# print(data_with_clusters)
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


kmeans.inertia_
# print(kmeans.inertia_)

wcss=[]

for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(x)

    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
print(wcss)


## Elbow Method

number_cluster=range(1,11)
plt.plot(number_cluster,wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares")

plt.show()


""" Categorical clustering """

#
# data_mapped=data.copy()
# data_mapped['Language']=data_mapped['Language'].map({'English':0,'French':1,"German":2})
# # print(data_mapped)
# print(data_mapped)
#
# x=data_mapped.iloc[:,1:4]
# kmeans=KMeans(3)
# kmeans.fit(x)
#
#
# identified_clusters=kmeans.fit_predict(x)
# # print(identified_clusters)
#
# data_with_clusters=data_mapped.copy()
# data_with_clusters['Cluster']=identified_clusters
# # print(data_with_clusters)
# plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.show()