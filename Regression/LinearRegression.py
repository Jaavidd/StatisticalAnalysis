import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()

    ## Predict GPA based on SAT
def p(d):
    print(d)

data=pd.read_csv("LinearReg.csv")


y=data['GPA']
x1=data['SAT']

plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
# plt.show()

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
p(result.summary())



x=sm.add_constant(x1.values)
results=sm.OLS(y,x).fit()

# print(results.summary())

plt.scatter(x1,y,c='blue')
yhat=0.0017*x1+0.275
fig=plt.plot(x1,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('size',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.show()