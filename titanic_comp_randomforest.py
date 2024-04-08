import pandas as pd
import numpy as np
from sklearn.svm._libsvm import fit

data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
test_ids=test["PassengerId"]

#First we just need to clean the train data for get the best outcome in the prediction

def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)  # We are deleting these columns because after we observe this data, the columns don't actually provide useful information
    cols = ["SibSp", "Parch", "Fare", "Age"]
    #We actually filling the missing data in this useful column by their column median
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data.Embarked.fillna("U",inplace=True)
    return data


data=clean(data) #We are cleaning the data (train set)
test=clean(test) #We are cleaning the test (test set)

#We are going to change the gender(string) column into binary using dummy

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
cols=["Sex","Embarked"]

for col in cols:
    data[col]=le.fit_transform(data[col])
    test[col]=le.transform(test[col])
    print(le.classes_)

# Import RandomForestClassifier from sklearn.ensemble module
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

Y=data["Survived"]                                  #Y contain the data of whether a person a person is survived or not(0 or 1)
X=data.drop("Survived", axis=1)              #X contain data (except the survived coloumn)

X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=42)  #splitting the data set into test set and train set
# Create and fit a RandomForestClassifier object
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, Y_train)
prediction=rfc.predict(np.array((X_val)))

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_val,prediction))#We are checking the accuracy of the model by comparing the outcome of prediction with the existing data
Submission_preds=rfc.predict(test)  #we containing the prediction of test in this variable
df=pd.DataFrame({"PassengerId":test_ids.values,
                                "Survived":Submission_preds,
                 })
df.to_csv("Submission2.csv",index=False )  #Transfering the above created data frame into new csv file(containing Survived and PassengerId)
