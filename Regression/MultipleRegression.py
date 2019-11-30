import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm
seaborn.set()

data=pd.read_csv("1.02. Multiple linear regression.csv")
# print(data)

y=data['GPA']
x1=data[['SAT','Rand 1,2,3']]

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()

# print(result.summary())

data=pd.read_csv("real_estate_price_size_year.csv")


y=data['price']
x1=data[['size','year']]

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary())