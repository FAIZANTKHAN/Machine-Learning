import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

d = pd.read_csv("hiring.csv")
d.experience = d.experience.fillna("zero")
d.experience = d.experience.apply(w2n.word_to_num)
import math
median_test_score = math.floor(d['test_score'].mean())
d['test_score'] = d['test_score'].fillna(median_test_score)

reg = linear_model.LinearRegression()
reg.fit(d[['experience','test_score','interview_score']],d['salary'])
reg.predict(np.array([[2,9,6]]))
reg.predict(np.array([[12,10,10]]))


