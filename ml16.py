#Hyper parameter tunning(Finding best model and hyper parameter tuning using GridSearchCv
#We are suing Iris dataset
import numpy as np
from sklearn import svm,datasets
iris=datasets.load_iris()

import pandas as pd
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#Changing the dataset into dataframe
df['flower']=iris.target
df['flower']=df['flower'].apply(lambda x:iris.target_names[x])
#Adding the another column with flowers name with correcponding to their repective data


#Approach 1:Use train_test_split and maually tune parameters  by trial and error
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)
model=svm.SVC(kernel="rbf",C=30,gamma='auto')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))  #93%

#Approach 2:Use K Fold Cross Validation
#Manually try suppling models with different parameters to cross_val_score function with  5 fold cross validation
from sklearn.model_selection import cross_val_score
#print(cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),iris.data,iris.target,cv=5))
#print(cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),iris.data,iris.target,cv=5))
#print(cross_val_score(svm.SVC(kernel='rbf',C=20,gamma='auto'),iris.data,iris.target,cv=5))

#Above approach is tiresome and manual,we can use for loop as as alternative

kernels=['rbf','linear']
C=[1,10,20]
avg_scores={}
for kval in kernels:
    for cval in C:
        cv_scores=cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),iris.data,iris.target,cv=5)
        avg_scores[kval+'_'+str(cval)]=np.average(cv_scores)
#print(avg_scores)


#Approach 3:Use GridSearchCV

from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)
#print(clf.cv_results_)  #Its is difficult to understand so i change that the results into dataframe

df=pd.DataFrame(clf.cv_results_)
print(df)

#print(df[['param_C','param_kernel','mean_test_score']])#For getting the at which parameter (C.kernel)what we are getting the the score

#Lets get the best parameter(C,kernel)
#print(clf.best_params_)  #C:1,kernel:rbf
#Lets get the best model score
#print(clf.best_score_)  #Score:98%

#Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters.
# This is useful when you have too many parameters to try and your training time is longer. It helps reduce the cost of computation

#Lets use RandomizedSearchCV(for less computational cost)

from sklearn.model_selection import RandomizedSearchCV
rs=RandomizedSearchCV(svm.SVC(gamma="auto"),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},
    cv=5,
    return_train_score=False,
    n_iter=2
)
rs.fit(iris.data,iris.target)
print(pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']])


#How about different models with different hyperparameters?

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores=[]


for model_name,mp in model_params.items():
        clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
        clf.fit(iris.data,iris.target)
        scores.append({
            'model':model_name,
            'best_score':clf.best_score_,
            'best_params':clf.best_params_
        })

df=pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df)
#Based on this we can that the SVM is efficient for this dataset
#With the model paarmeter=C:1 ,kernel:'rbf