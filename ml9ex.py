import pandas as pd
df=pd.read_csv("titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
inputs=df.drop('Survived',axis='columns')    #Features that are choose as independent
target=df.Survived   #Features that are choose as dependent
inputs.Sex=inputs.Sex.map({'male':1,'female':2})  #Mapping the gender as 1 for male and 2 for female because they are ordinal categorial
#We clearly see that age has some NaN value so fill it with mean of the whole age column
inputs.Age=inputs.Age.fillna(inputs.Age.mean())

#Let split the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)


#let use decision tree to make the prediction of the survival
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))   #Score is 78%