import pandas as pd
df=pd.read_csv("carprices.csv")

import matplotlib.pyplot as plt
plt.scatter(df['Mileage'],df['Sell_Price'])
X=df[['Mileage','Age']]
y=df['Sell_Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#20% of the data is contain by test set whereas 80% of data is contain by train set

from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,y_train)  #Training the data
print(clf.predict(X_test))
print(clf.score(X_test,y_test)) #to check accuracy of the model




