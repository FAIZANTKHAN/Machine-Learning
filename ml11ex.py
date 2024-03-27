#Random Forest Exercise
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()

df=pd.DataFrame(iris.data, columns=iris.feature_names)
#Change the whole data into data frame
df['target']=iris.target  #Add the new column for target(which contain the flower names versicolor,setosa,verginica)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()