#Principle Component Analysis
#It is used to reduce the no.of columns(features)
from sklearn.datasets import load_digits
import pandas as pd

dataset=load_digits()
#print(dataset.keys())

#print(dataset.data.shape)  #(1797,64)


#print(dataset.data[0])
#print(dataset.data[0].reshape(8,8))  #for reshaping the data

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))  #For printing the Oth element
plt.matshow(dataset.data[9].reshape(8,8)) #For printing the 9th element
#plt.show()


#print(dataset.target[:5])  #For printing the target of the data from 0 to 5(0,1,2,3,4)
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
#Changing the whole dataset into the dataframe(df)


#Lets see the target dataset
#print(dataset.target)  #0 to 9

#Lets see the basic statistics of our main dataset
#print(df.describe())


X=df
y=dataset.target

from sklearn.preprocessing import StandardScaler
#Lets do the preprocessing of the data (scaling)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)  #Scaling is done on the X(dataset)

#Lets split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)




#Applying Logistic Regression

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)  #Training our trainig set
print(model.score(X_test,y_test))  #Score of the Model  (96%)

#Lets use PCA to reduce dimensions
#The original dimension of the dataset is (1797,64)

from sklearn.decomposition import PCA

pca=PCA(0.95)  #This means we going to use the 95% of data which have more imoact on the model while predicting

x_pca=pca.fit_transform(X)
#print(x_pca.shape)  #After using this we get(1797,29)
#print(pca.explained_variance_ratio_)  #For knowing the variance
print(pca.n_components_)  #To know the no.of  features(columns)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(model.score(X_test_pca, y_test))  #After using the socre is still 96% but you see there is way difference in the computational power because we  deleted lots of columns
#But this doesn't mean that it worked on every dataset works so well ,sometime the accuracy may degrade

#Lets see what will happen if i only select 2 components(Most Impactful components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)  #(1797,2)

#Lets see the variance Ratio
print(pca.explained_variance_ratio_)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(model.score(X_test_pca, y_test))


#We get less accuancy (~60%) as using only 2 components did not retain much of the feature information.
# However in real life you will find many cases where using 2 or few PCA components can still give you a pretty good accuracy

