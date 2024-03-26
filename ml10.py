import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#Support Vector Machine

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

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train) #Training the data using support vector machine model

print(model.score(X_test,y_test))   #Checking the score i.r the accuracy of the model by applying the model into test data set
#so after appllying this we came to know that that the model score is 96%