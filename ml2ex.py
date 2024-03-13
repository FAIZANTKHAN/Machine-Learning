import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

df=pd.read_csv("canada_per_capita_income.csv")
plt.scatter(df.pcincome,df.year, color="green",marker="+")
plt.title("Per Capita Income Vs Year")
plt.xlabel("Per Capita Income")
plt.ylabel("Year")
reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.pcincome)
print(reg.predict(np.array( [ [2020]] ))) #So using I can predict the per capita income of  year 2020
print(reg.predict(np.array( [ [2017]] )))#Same way the year 2017
m, b = np.polyfit(df.pcincome, df.year, 1)
# obtain the slope and intercept of the regression line
plt.plot(df.pcincome, m*df.pcincome + b, color="red") #Plotting the line of regression on the graph
plt.show()