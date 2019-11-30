from cmath import sqrt

import pandas as pd




data=pd.read_csv("CInterval.csv")
# print(data)

values=data['Difference']

sum=0
for i in values:
    sum+=i

mean=sum/len(data['Difference'])
print(mean)

std_sum=0
for i in values:
    std_sum+=pow(i-mean,2)

stdDeviation=sqrt(float("{0:.3f}".format(std_sum/(len(data["Difference"])-1))))

print(stdDeviation)

low=mean-2.26*(stdDeviation/sqrt(len(data["Difference"])))
high=mean+2.26*(stdDeviation/sqrt(len(data["Difference"])))
# low=float(low)

print("[",low,",",high,"]")