#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('/content/drive/MyDrive/유데미/machine_learning/section6_회귀/3_다항회귀/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(dataset)
print(X.shape)
print(y.shape)
print()

#taining the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
print(X.shape)

#training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
print(X_poly.shape)
print()

lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

#visualizing the linear regression results
# plt.scatter(X, y, color='red')
# plt.plot(X, lr.predict(X)) #예측된 연봉
# plt.show()
print(lr_2.predict(X_poly))
#visualizing the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lr_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lr_2.predict(pf.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predict
print(lr.predict([[6.5]]))
print(lr_2.predict(pf.transform([[6.5]])))

