#Support Vector Machine Exercise

import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
#Lets change these data into dataframe
df=pd.DataFrame(digits.data,digits.target)


#Now we are going to add the column of target into main dataframe
df['target']=digits.target

#Lets split the data into test and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop("target",axis="columns"),df.target,test_size=0.2)


#Now we are going to use the SVC
#RBF kernel

from sklearn.svm import SVC
rbf_model=SVC(kernel='rbf')
rbf_model.fit(X_train,y_train)
print(rbf_model.score(X_test,y_test))   #Score is 98%

#Linear Model

from sklearn.svm import SVC
linear_model=SVC(kernel='linear')
linear_model.fit(X_train,y_train)

print(linear_model.score(X_test,y_test))  #Score is 96%
