import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans



# Load the data
raw_data = pd.read_csv('Categorical.csv')

data = raw_data.copy()


data_mapped = data.copy()
data_mapped['continent'] = data_mapped['continent'].map({'North America':0,'Europe':1,'Asia':2,'Africa':3,'South America':4, 'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})
# data_mapped
x = data_mapped.iloc[:,3:4]

kmeans = KMeans(8)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)

data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters

plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()