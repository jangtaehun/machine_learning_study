#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#앙상블 학습의 한 버전
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("/content/drive/MyDrive/유데미/study/section6_회귀/6_랜덤포레스트/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
rand = RandomForestRegressor(n_estimators=10, random_state=0) #트리의 수
rand.fit(X, y)
print(rand.predict([[6.5]])) #predict는 2차원 배열을 입력해야 한다.

plt.scatter(X, y)
plt.plot(X, rand.predict(X))
plt.show


# In[ ]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red') #svr예측으로 교체
plt.plot(X_grid, rand.predict(X_grid))
plt.show()

