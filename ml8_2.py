import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()

print(digits.data[0])  #This means the data is stored in the form of  array
#To change it in the picture
plt.gray()
#plt.matshow(np.reshape(digits.data[0], (8, 8))) #Reshape the 1D array into a 2D array



#Now we are going to split the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.2)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(digits.data)
y = digits.target

# Create and fit the logistic regression model
# Increase the max_iter and choose a different solver
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X, y)

# Plot the image of a digit
plt.gray()
plt.matshow(np.reshape(digits.data[67], (8, 8))) #Remove the outer plt.matshow call
plt.show()

#print(digits.target[67]) #So at digits.data 67 is 6
#Lets predict through our model
#model.predict(np.array(digits.data[67]) ,(8,8)) #So its showing exactly same

#My model is 96% accurate but i want to know where my model fails(means at which data)

#So we are going to use confusion matrix

y_predicted=model.predict(np.array(X_test) )
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()