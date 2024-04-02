#K nearest neighbour classification

#Starting code part is from SVM
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt



iris = load_iris()
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Creating a dataframe that contains the data of the iris dataset and then adding another column named 'target'
df['target'] = iris.target
print(iris.target_names)  # To know the target names (i.e setosa, versicolor, virginica)

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])  # Adding the new parameter

df0 = df[df.target == 0]  # Setosa
df1 = df[df.target == 1]  # Versicolor

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], c='g', marker='+', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], c='b', marker='.', label='Versicolor')

plt.legend()  # Adding a legend to differentiate between the two classes
plt.show()

from sklearn.model_selection import train_test_split
X=df.drop(['target','flower_name'],axis='columns')
y = df.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #We are splitting the data into 2 part 80 %part of data is in training part and other 20% is in test part

#Lets Use K nearest neighbour

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)  #n_neighbors means that how many nearest neighbors we are considering in this model
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))  #93%

#Now we try to plot the confusion matrix to know that at point of the data it doesn't able to predict well
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)  #storing the predicted value of  this model(on X_test)in y_pred
cm=confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))  #To know the detail about the model(support,accuracy,f1-score,weighted,weighted avg,etc)