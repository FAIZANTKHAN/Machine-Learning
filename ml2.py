import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Linear Reagression In Single Variable

df=pd.read_csv("ml2.csv")
plt.scatter(df.Area, df.price ,color='red',marker='+')
plt.xlabel('area(sqr ft')
plt.ylabel('price(US$')
reg=linear_model.LinearRegression()  #We are using linear regression which is extracted from the linear_model
reg.fit(df[['Area']],df.price)  #We have to find the price according to the area
print(reg.predict(np.array( [ [3300]] )))  #It accepts 2D array


#Proving this conceptusing y=mx+b concept
#Compare to this we have to find price=m*area+b

m=reg.coef_
b=reg.intercept_

print(m*3300+b)  #Hence Proved