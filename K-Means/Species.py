import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
from sklearn.cluster import KMeans

data=pd.read_csv("iris-dataset.csv")
print(data)
plt.scatter(data['sepal_length'],data['sepal_width'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.xlabel('length')
plt.ylabel('width')
# plt.show()

x=data.copy()
kmeans=KMeans(2)
kmeans.fit(x)

clusters=data.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)
plt.scatter(clusters['sepal_length'],clusters['sepal_width'],c=clusters['cluster_pred'],cmap='rainbow')
# plt.show()

from sklearn import preprocessing

x_scaled=preprocessing.scale(data)
# print(x_scaled)

kmeans_scaled=KMeans(5)

kmeans_scaled.fit(x_scaled)

clusters_scaled=data.copy()
clusters_scaled['cluster_pred']=kmeans_scaled.fit_predict(x_scaled)

plt.scatter(clusters_scaled['sepal_length'],clusters_scaled['sepal_width'],c=clusters_scaled['cluster_pred'],cmap='rainbow')
plt.show()

wcss=[]

for i in range(1,10):
    kmeans=KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1,10),wcss)
plt.title("Elbow")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()
plt.close()

real_data=pd.read_csv('iris-with-answers.csv')
real_data['species'].unique()
real_data['species']=real_data['species'].map({'setos':0,'versicolor': 1,'virginica':2})
real_data.head()

plt.scatter(real_data['sepal_length'],real_data['sepal_width'],c=real_data['species'],cmap='rainbow')
plt.show()