
data22=read_csv("path that contain sales forecast data")
print(data22.describe())

print(data22.isnull().sum())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

reg_model=LinearRegression()
x=data22['BasePrice'].values.reshape(-1,1)
y=data22['SalesPrice'].values.reshape(-1,1)

#80% training data and 20% test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

reg_model.fit(x_train,y_train)

#for retrieve intercept
print(reg_model.intercept_)
#for retrieve slope
print(reg_model.coef_)

y_pred=reg_model.predict(x_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

#random prediction
a=pd.DataFrame({
        'baseprice':[218.77,87.28,90.64]},
         index=[1,2,3])
b=a.values.reshape(-1,1)
z=reg_model.predict(b)
z

#visualize comparison of y_test and y_pred as bar graph
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,4))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#lets plot the straight line with the test data
plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

rmse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(rmse)
print(r2)

#if independent values are more than one then we use multilinear regression for that we have only one change
#X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
#y = dataset['quality'].values

