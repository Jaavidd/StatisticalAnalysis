import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

data=pd.read_csv("andard-normal-distribution-exercise.csv")
# print(len(data['Original dataset']))
sum=0

for i in data['Original dataset']:
    sum=sum+i

mean=sum/len(data['Original dataset'])
mean = float("{0:.2f}".format(mean))

print("mean ",mean)
var_sum=0
for i in data['Original dataset']:
    var_sum=var_sum+pow(i-mean,2)

variance=var_sum/len(data['Original dataset'])

print("variance ",variance)
standard_deviation=math.sqrt(variance)
print(standard_deviation)

# plt.plot(data['Original dataset'],stats.norm.pdf(data['Original dataset']),mean,standard_deviation)
x=np.linspace(mean-3*standard_deviation,mean+3*standard_deviation,100)
plt.plot(x,stats.norm.pdf(x,mean,standard_deviation))
plt.show()

"""Standardization"""

new_data=[]
for i in data['Original dataset']:
    new_data.append(float("{0:.2f}".format((i-mean)/standard_deviation)))
print(new_data)

sum=0
for i in new_data:
    sum=sum+i

standardize_mean=sum/len(data['Original dataset'])
print(float("{0:.2f}".format(sum/len(data['Original dataset']))))

x=np.linspace(standardize_mean-3*1,standardize_mean+3*1,100)
plt.plot(x,stats.norm.pdf(x,standardize_mean,1))
plt.show()