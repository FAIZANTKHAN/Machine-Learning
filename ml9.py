import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier\


df = pd.read_csv("salaries.csv")

# Separate input features and target variable
inputs = df.drop("salary_more_then_100k", axis='columns')
target = df['salary_more_then_100k']

# Label encode categorical columns
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['Company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Apply one-hot encoding to the categorical columns
ohe = OneHotEncoder()
inputs_encoded = ohe.fit_transform(inputs[['Company_n', 'job_n', 'degree_n']]).toarray()

# Concatenate the encoded columns with the rest of the data
inputs = pd.concat([inputs, pd.DataFrame(inputs_encoded)], axis=1)

# Drop the original categorical columns
inputs.drop(['company', 'job', 'degree', 'Company_n', 'job_n', 'degree_n'], axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Using decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

import numpy as np

# Assuming 'model' is the trained DecisionTreeClassifier

# Predictions for the input [2, 2, 1]
prediction_1 = model.predict(np.array([[2, 2, 1]]))
print("Prediction for [2, 2, 1]:", prediction_1)

# Predictions for the input [2, 0, 1]
prediction_2 = model.predict(np.array([[2, 0, 1]]))
print("Prediction for [2, 0, 1]:", prediction_2)
