#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("/content/drive/MyDrive/유데미/study/section6_회귀/5_의사결정트리/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#의사 결정 나무 회귀, 랜덤 포레스트 회귀는 스케일링 필요 없다.
#연속적으로 분할된 데이터에서 나온 결과

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=0)
clf.fit(X, y)
clf.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red') #svr예측으로 교체
plt.plot(X_grid, clf.predict(X_grid))

