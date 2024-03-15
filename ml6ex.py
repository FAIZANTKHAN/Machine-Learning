import pandas as pd
import numpy as np
from pandas import get_dummies

df=pd.read_csv("carprices.csv")
dummies=get_dummies(df.Car_Model)   #Creating binary form of the Car_Model column
merge=pd.concat([df,dummies],axis="columns")
final=merge.drop(['Car_Model','Mercedez Benz C class'],axis='columns')




X=final.drop('Sell_Price',axis='columns')  #X contain all the data from final(only dropping the Sell Price column
y=final['Sell_Price']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model.score(X,y))  #to check the accuracy of the model we built
print(model.predict([[45000,4,0,0]]))  #Price of mercedez benz class c that is 4 yr old with mileage 45000
print(model.predict([[86000,7,0,1]])) #Price of BMW X5 that is 7 yr old with mileage 86000