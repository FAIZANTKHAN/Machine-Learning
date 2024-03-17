import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Logistic Regression
#We have to build a machine learning model that can predict whether a person is going to buy insurance or not
df=pd.read_csv("insurance_data.csv")

#plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
#plt.show()


#split the data into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(model.predict_proba(X_test))
print(model.score(X_test,y_test))

