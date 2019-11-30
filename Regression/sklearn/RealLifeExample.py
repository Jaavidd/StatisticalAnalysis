import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import seaborn as sns
# sns.set()


data=pd.read_csv('1.04. Real-life example.csv')

pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',200)
# print(data)

data=data.drop(['Model'],axis=1)
# print(data.describe(include='all'))

data_no_mv=data.dropna(axis=0)

# plt.show(sns.distplot(data_no_mv['Price']))

""""""""" eliminate Outliers """""
""""""""""DATA CLEANING"""""""""""


queue=data_no_mv['Price'].quantile(0.99)
data_1=data_no_mv[data_no_mv['Price']<queue]


q=data_1['Mileage'].quantile(0.99)
data_2=data_1[data_1['Mileage']<q]

data_3=data_2[data_2['EngineV']<6.5]

q=data_3['Year'].quantile(0.01)
data_4=data_3[data_3['Year']>q]

# plt.show(sns.distplot(data_4['Year']))

"""""""""DATA CLEANING SUCCESFULLY END"""""""""

Data_cleaned=data_4.reset_index(drop=True)
# print(Data_cleaned)

"""""CHECKING THE OLS ASSUMPTION"""""

f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True, figsize= (15,3) )
ax1.scatter(Data_cleaned['Year'],Data_cleaned['Price'])
ax1.set_title("Price and Year")

ax2.scatter(Data_cleaned['EngineV'],Data_cleaned['Price'])
ax2.set_title("Price and EngineV")

ax3.scatter(Data_cleaned['Mileage'],Data_cleaned['Price'])
ax3.set_title("Price and Mileage")

# plt.show()

""""Since Price creates exponential regression,we represent it as a logarithmic"""

price_log=np.log(Data_cleaned['Price'])
Data_cleaned['Log Price']=price_log
Data_cleaned=Data_cleaned.drop(['Price'],axis=1)
# print(Data_cleaned)


"""""""""""""""""MULTICOLINEARITY"""""""""""
from statsmodels.stats.outliers_influence import variance_inflation_factor
varibales=Data_cleaned[['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(varibales.values,i)for i in range(varibales.shape[1])]
vif["features"]=varibales.columns
# print(vif)
"""AS We seef from VIF table that Year is too corellated with other 2 varibales and we drop Year """

data_no_multicollinearity=Data_cleaned.drop(['Year'],axis=1)

""""CREAT DUMMY VARIABLES"""
data_with_dummies=pd.get_dummies(data_no_multicollinearity,drop_first=True)


columns=['Log Price', 'Mileage' ,'EngineV' , 'Brand_BMW' ,'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen',
 'Body_hatch' ,'Body_other', 'Body_sedan', 'Body_vagon' ,'Body_van',
 'Engine Type_Gas', 'Engine Type_Other' ,'Engine Type_Petrol',
 'Registration_yes']

Data_preprocessed=data_with_dummies[columns]
# print(Data_preprocessed.head())


"""""LINEAR REGRESSION MODEL"""
target=Data_preprocessed['Log Price']
inputs=Data_preprocessed.drop(['Log Price'],axis=1)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(inputs)
inputs_scaled=scaler.transform(inputs)


""""TRAN TEST SPLIT"""
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(inputs_scaled,target,test_size=0.2,random_state=365)

reg=LinearRegression()
reg.fit(x_train,y_train)

y_hat=reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel("Targets(y_hat)",size=18)
plt.ylabel("Predictions(y_hat)",size=18)
plt.xlim(6,13)
plt.ylim(6,13)

# plt.show()
""""""""""TESTING"""""""""
y_hat_test=reg.predict(x_test)
plt.scatter(y_test,y_hat_test,alpha=0.3)

plt.xlabel("Targets(y_test)",size=18)
plt.ylabel("Predictions(y_hat_test)",size=18)
plt.xlim(6,13)
plt.ylim(6,13)
# plt.show()

df_pf=pd.DataFrame(np.exp(y_hat_test),columns=['Predictions'])
# print(df_pf.head())
y_test=y_test.reset_index(drop=True)
df_pf['Target']=np.exp(y_test)
df_pf['Residual']=df_pf['Target']-df_pf['Predictions']
df_pf['Difference%']=np.abs(df_pf['Residual']/df_pf['Target']*100)
pd.options.display.max_rows=999
pd.set_option('display.float_format',lambda x: '%.2f' % x)
df_pf=df_pf.sort_values(by=['Difference%'])

print(df_pf)
