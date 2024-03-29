import pandas as pd
import numpy as np
df=pd.read_csv("spam.csv")
#print(df.groupby("Category").describe())  to see the count of unique an top freq form both spam and ham

df["spam"]=df["Category"].apply(lambda X: 1 if X=="spam" else 0)  #To add a column that have the mapping of the spam to 1 and ham to 0
#Split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.2)

#Now we are going to change the word into matric mapping using count vectorizer

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create a CountVectorizer object
count_vect = CountVectorizer()

# Transform the training data into feature matrix
X_train_count = count_vect.fit_transform(X_train)

# Create and fit a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Transform the test data into feature matrix
X_test_count = count_vect.transform(X_test)

# Print the model's accuracy score on the test data
print(model.score(X_test_count, y_test))
#Score is 98%

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('Vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
#By this we are storing a model that can do both the Multimonial Naive Bayes  as well as CountVectorizer
#Then by this model we try to fit the model

clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


