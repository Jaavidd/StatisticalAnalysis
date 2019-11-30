import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',200)

data=pd.read_csv("2.01. Admittance.csv")
data['Admitted']=data['Admitted'].map({'Yes':1,'No':0})

y=data['Admitted']
x1=data['SAT']



x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
result_log=reg_log.fit()
print(result_log.summary())