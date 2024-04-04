#Naive Bayes Part 2 Exercise
from sklearn import datasets
wine=datasets.load_wine()

#print(wine.data[0:2])
#print(wine.feature_names) #Feature names
#print(wine.target_names)  #Class 0,class 1,class 2

#Changing the dataset into dataframe
import pandas as pd
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target']=wine.target  #Adding a new column target containing target class 0,1,2.
#print(df[50:70])
#Splitting the data into training and testing data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3,random_state=100)


#Using Gaussian And Multimonial Naive Bayes for training our data

#lets first try gaussian naive bayes

from sklearn.naive_bayes import GaussianNB,MultinomialNB
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test)) #Checking the score of the model
#After using Gaussian NB,the score is 100%

#lets use multimonial naive bayes
from sklearn.naive_bayes import MultinomialNB
mn=MultinomialNB()
mn.fit(X_train,y_train)
print(mn.score(X_test,y_test)) #Checking the score of the model
#After using Gaussian NB,the score is 77%