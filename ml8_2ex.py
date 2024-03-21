# Import the libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # Labels

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the logistic regression model
log_reg = LogisticRegression(max_iter=1000) # Increase the number of iterations
log_reg.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = log_reg.predict(X_test.reshape(30,4)) # Reshape the array

# Compute the accuracy score
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Predict some samples from the test set
samples = X_test[[0, 10, 20]] # Select three samples
predictions = log_reg.predict(samples.reshape(-1, 1)) # Reshape the array
print(f"Predictions: {predictions}")
