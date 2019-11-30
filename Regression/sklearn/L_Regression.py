import numpy as np
import pandas as pd
import seaborn
seaborn.set()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# data=pd.read_csv("1.02. Multiple linear regression.csv")
# # print(data)
# x=data[['SAT','Rand 1,2,3']]
# y=data['GPA']
# # x_matrix=x.values.reshape(-1,1)
#
# regressiong=LinearRegression()
# #
# regressiong.fit(x,y)

def R_Adjust(x,y):
   R_Squared= reg.score(x,y)
   Number_Of_Observation=x.shape[0]

   R_Adjusted = 1 - (1 - R_Squared) * (Number_Of_Observation - 1) / (Number_Of_Observation - x.shape[1] - 1)
   return R_Adjusted



# print(R_Adjust(x,y))



# f_regression(x,y)
# # print(f_regression(x,y)[1])
# p_values=f_regression(x,y)[1]
# p_values=p_values.round(3)

# print(p_values)


data=pd.read_csv("real_estate_price_size_year.csv")
y=data['price']
x=data[['size','year']]
x_mat=x.values.reshape(-1,1)

reg=LinearRegression()
reg.fit(x,y)

print(R_Adjust(x,y))
pval=f_regression(x,y)

newD=pd.DataFrame(data=x.columns.values,columns=['Feature'])
newD['coef']=reg.coef_
newD['p-values']=pval[1].round(3)
print(newD)


#
#
# print(new)
#
# plt.scatter(x,y)
# equat=x*reg.coef_ + reg.intercept_
# flg=plt.plot(x,equat,lw=4,c='orange',label='regression line')
# plt.xlabel('size',fontsize=20)
# plt.ylabel('price',fontsize=20)
# plt.show()
#


