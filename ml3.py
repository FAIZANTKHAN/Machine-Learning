import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

df=pd.read_csv("homeprices.csv")
#first we have to fill the null value
median_bedrooms=math.floor(df.bedrooms.median())#we store the floor value of median of no.of bedrooms in the variable median _bedrooms
print(median_bedrooms)
#we should update the dataframe after fll the NaN
df.bedrooms=df.bedrooms.fillna(median_bedrooms)

#------Now our data is cleaned ----------
#-----Next step is predicting the price-------

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)  #We are going to prdict the price(dependent variable) by the use of  independent variable
#and this variables(independent and dependent variable) together called features
print(reg.coef_ )#Coefficient (here there is 3 variable so 3 coefficient will be here this is totally depend on the no.of independent variable
print(reg.intercept_)#intercept
print(reg.predict(np.array([[3000,3,40]]))) #We are trying to predict the price of a room with 3000 sq ft area,3 bedroom,and 40 yr old room
#we just calculated by(m1*area+m2*bedrooms+m3*age+b)
print(reg.predict(np.array([[2500,4,5]])))

