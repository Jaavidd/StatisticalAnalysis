import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import statsmodels.api as sm

pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows',2000)

data=pd.read_csv('Example-bank-data.csv')

data=data.drop(['Unnamed: 0'],axis=1)
data['y']=data['y'].map({'yes':1,'no':0})
# print(data)

y=data['y']
x1=data['duration']

x=sm.add_constant(x1)
log=sm.Logit(y,x)
result=log.fit()
print(result.summary())

plt.scatter(x1,y,color='C0')
plt.show()