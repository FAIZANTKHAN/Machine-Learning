#K-means Clustering
#New Copy (Continue)


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import  matplotlib.pyplot as plt

df=pd.read_csv("income.csv")
#plt.scatter(df.Age,df["Income($)"])
#plt.xlabel('Age')
#plt.ylabel('Income($)')
#plt.show()

km=KMeans(n_clusters=3,n_init=10)  #Using KMeans and  using k=3
y_predicted=km.fit(df[['Age','Income($)']]) #Fitting the the Age and Income Using KMaens

df['cluster']=y_predicted   #Adding a new column which tells which age vs income lie in which cluster
#print(km.cluster_centers_)  #For printing the positions of centriod

#df1 = df[df.cluster==0]
#df2 = df[df.cluster==1]
#df3 = df[df.cluster==2]
#plt.scatter(df1.Age,df1['Income($)'],color='green')
#plt.scatter(df2.Age,df2['Income($)'],color='red')
#plt.scatter(df3.Age,df3['Income($)'],color='black')
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')  #For showing the centroids
#plt.xlabel('Age')
#plt.ylabel('Income ($)')
#plt.legend()
#plt.show()

#Upto this we are using normal data withouir preprocessing and we can clearly see that the centroid is shifted from the actual cluster
#Lets do the same thing but first use the preprocessing

##Preprocessing using min max scaler
scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
#Minimizing the valuie of income by using MinMaxScaler for better calculation
km = KMeans(n_clusters=3,n_init=10)
y_predicted = km.fit_predict(df[['Age','Income($)']])
df['cluster']=y_predicted
km.cluster_centers_


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

#Lets draw the Elbow Plot(Its use to find the optimal value of K)

sse=[] #Sum of Squared Error
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_init=10)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_rng,sse)
plt.show()