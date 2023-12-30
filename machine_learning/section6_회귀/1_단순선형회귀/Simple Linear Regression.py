#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#최소 제곱법
#y - y^ = 잔차 -> 잔차 / 잔차 제곱의 합이 최소화 => 최소 제곱법


# #Simple Linear Regression
# 단순 선형 회귀에서 상수 = 선이 수직 축과 교차하는 부분
# 종속 변수 = 경력연수에 따라 급여가 어떻게 변하는지와 같이 설명하려는 대상

# Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset

# In[11]:


dataset = pd.read_csv('./Salary_Data.csv')
#머신 러닝 모델 훈련 데이터 세트에는 특성 / 종속 변수 백터가 있다.
#특성 열을 사용해 종속 변수 예측 -> 종속 변수는 마지막 열

X = dataset.iloc[:, :-1].values #values = 데이터를 넘파이 배열로 추출
y = dataset.iloc[:, -1].values
print(X.shape)
print(y.shape)
print(X)
print(y)
print(dataset)


# In[7]:


#결측치 -> 평균으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#missing_values=np.nan 모든 결측값을 대체한다. / strategy='mean' 평균으로 대체한다.
imputer.fit(X[:,:]) #숫자값을 가지는 열만 전달해야 한다. 어디에 결측값이 있을지 모르니 모든 숫자형 열 선택
X[:,:] = imputer.transform(X[:,:])
print(X.shape)


# Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1) #권장 훈련 세트 80%, 테스트 세트 20%


# Training the simple Linear model on the Training set

# In[9]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train) #X_train - 특성 = 훈련 세트의 독립 변수 / y_train - 훈현 세트의 종속 변수 벡터


# Predicting the Test set results

# In[12]:


y_pred = lr.predict(X_test) #특성 입력 -> y_test= 실제 임금 / y_pred= 예측 임금
print(X_train.shape)
print(y_train.shape)
print(X_train)
print(y_train)


# Visualising the Training set results

# In[ ]:


plt.scatter(X_train,y_train, color='red') #x축은 근무 횟수 / y축은 급여
plt.plot(X_train, lr.predict(X_train), color='blue') #만들고자 하는 그래프의 y좌표 입력 = 훈련 세트의 예측 급여
#예측값에 해당하는 점의 좌표 또는 선의 점을 입력 -> 훈련 세트의 결과를 시각화 / y = 훈련 세트의 예측 급여가 포함된 벡터
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set result

# In[ ]:


plt.scatter(X_test,y_test, color='red') #x축은 근무 횟수 / y축은 급여
plt.plot(X_test, lr.predict(X_test), color='blue') #만들고자 하는 그래프의 y좌표 입력 = 훈련 세트의 예측 급여
#예측값에 해당하는 점의 좌표 또는 선의 점을 입력 -> 훈련 세트의 결과를 시각화 / y = 훈련 세트의 예측 급여가 포함된 벡터
#test set와 train set의 예측과 같은 회귀선 -> 굳이 X_train, lr.predicr(X_train)을 X_test로 바꾸지 않아도 된다
#단순 선형 회귀 모델의 회귀선은 고유한 식에서 도출됨으로 회귀선은 같다.
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(lr.score(X_test, y_test)) #테스트 세트를 통한 모델 평가
print(lr.score(X_train, y_train)) #훈련 세트를 통한 모델 평가
print(lr.predict([[12]])) #경력이 12년인 직원의 급여 예측
print(X_test) #[[12]]→2D array


# In[ ]:


#최종 회귀 방정식 y = b0 + b1x를 구하는 방법
print(lr.coef_) #기울기 theta1
print(lr.intercept_)# y절편 theta0
#Salary = 9345.94 × YearsExperience + 26816.19

