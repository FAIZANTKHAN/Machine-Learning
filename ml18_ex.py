#KNN Classification
import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
#print(digits.target)  #So the targets ranges from 1 to 9
#print(dir(digits))  #show the what kind of columnsdoes this dataset will contain
#print(digits.target_names) #0 to 9


#Lets change the whole dataset into dataframe
df=pd.DataFrame(digits.data,digits.target)


#Add another column named as target
df['target']=digits.target


#Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis='columns'),df.target,test_size=0.3,random_state=30)

#Create KNN classifier

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))  #98%

#Lets create confusion matrix

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
