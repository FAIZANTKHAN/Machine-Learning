import pandas as pd
import numpy as np
from pandas import get_dummies

df=pd.read_csv("homepricesdummy.csv")
dummies=get_dummies(df.town)#it will create a binary form representation of town
#now we have to merge/concate the both the df and the dummies
merge=pd.concat([df,dummies],axis="columns")#concate both of them column wise


#Now we are going to drop the two column (1st one is town column and 2nd is one of the dummy variable column we just creates reason is mentioned in copy (dummy trap)
final=merge.drop(['town','west windsor'],axis='columns')  #nstead of west winsor you can delete anyone


from sklearn.linear_model import LinearRegression
model=LinearRegression()  #we create a object in which we save module LinearReagression
X=final.drop('price',axis='columns')  #we drop the price column and put into the X object
y=final.price #we put Price column in y object

model.fit(X,y)  #we are applying LinearRegression to X,y (X contain independent variable and y contain dependent variable
#Now we are going to predict the price of 2800 sq ft area in robbinsville
                #print(np.array(model.predict([[2800,0,1]])))
#Now we are going to predict the price of 3400 sq ft area in west windson
            #print(np.array(model.predict([[3400,0,0]])))


#Now we are going to use ONE HOT  ENCODING

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=df
le.fit_transform(dfle.town)  # it will create this [0 0 0 0 0 2 2 2 2 1 1 1 1] according to the town
dfle.town=le.fit_transform(dfle.town)
X=dfle[['town','area']].values  #You will get 2D dataframe(using values)
y=dfle.price      #In X we put the independent values whereas in the y we put the dependent

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories=[0]) # or specify the categories manually
X = ohe.fit_transform(X).toarray() # assign to a new variable
X=X[:1:]

model.fit(X,y)
print(np.array(model.predict([[1,0,2800]])))


