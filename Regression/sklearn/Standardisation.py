import numpy as np
import pandas as pd
import seaborn
seaborn.set()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

data=pd.read_csv("1.02. Multiple linear regression.csv")
# print(data)
x=data[['SAT','Rand 1,2,3']]
y=data['GPA']

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)
reg=LinearRegression()
reg.fit(x_scaled,y)

reg_summary=pd.DataFrame([['Bias'],['SAT'],['Rand']],columns=['Features'])
reg_summary['Weights']=reg.intercept_,reg.coef_[0],reg.coef_[1]
print(reg_summary)


new_data=scaler.transform([[1700,2]])
print(reg.predict(new_data))