#K Means Clustering
#1.Use iris flower dataset from sklearn library and try to form clusters of flowers using petal width and length features. Drop other two features for simplicity.
#2.Figure out if any preprocessing such as scaling would help here
#3.Draw elbow plot and from that figure out optimal value of k

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)  #Changing the whole  datset into dataframe

#print(df.head())
#adding the target column in the dataframe as flower
df['flower']=iris.target
#print(df.head())

#We are going to drop all the column except th petal length and petal width(you can keep all the feature we just want to make it simple
df.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'],axis='columns',inplace=True)
#print(df.head(5))

km=KMeans(n_clusters=3)#We are going to use Kmeans in which we are going to use 3 clusters (n_clusters=3)
yp=km.fit_predict(df)
print(yp)  #Fit the predicted data using kmeans

df['cluster']=yp#Adding an another column in which we put the predicted cluster after using kmeans
print(df.head(4))

print(df.cluster.unique())  #Printing the all th eunique value in the column cluster
#seperating the df  on the basis of the clusters

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]


#Plotting the graph for the clusters
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='green')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
plt.show()


#Lets find the optimal value of the K
#Using Elbow Plot

sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

