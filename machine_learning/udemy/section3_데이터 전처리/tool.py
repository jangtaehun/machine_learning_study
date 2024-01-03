# #Data Preprocessing Tools

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('./Data.csv')
#머신 러닝 모델 훈련 데이터 세트에는 특성 / 종속 변수 백터가 있다.
#특성 열을 사용해 종속 변수 예측 -> 종속 변수는 마지막 열

X = dataset.iloc[:, :-1].values #values = 데이터를 넘파이 배열로 추출
y = dataset.iloc[:, -1].values
print(X)
print(y)

# 
# 1.   Taking care of missing data
# 2.   Encoding categorical data
# 3. Encoding the Dependent Variable

# In[ ]:
#결측치 -> 평균으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #missing_values=np.nan 모든 결측값을 대체한다. / strategy='mean' 평균으로 대체한다.
imputer.fit(X[:,1:3]) #숫자값을 가지는 열만 전달해야 한다. 어디에 결측값이 있을지 모르니 모든 숫자형 열 선택
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)


#원핫 인코딩 / 한 특성이 여러 개의 범주를 가질 때
from sklearn.compose import ColumnTransformer #ColumnTransformer에는 fit_transform 메서드가 있어서 한 번에 처리할 수 있다.
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0] )], remainder='passthrough')
#transformer에는 세 가지 명시 -> 변환 유형인 인코딩, 인코딩 유형인 원핫 인코딩, 열의 인덱스 []안에 ()에 튜플 형태로 입력
#remainder은 나머지 열은 어떻게 처리할지를 명시 'passthrough' 나머지 열 변환하지 않는다.
X = ct.fit_transform(X)
print(X)


#레이블 인코딩 / 한 특성이 두 개의 범주를 가질 때
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# In[ ]:

# Splitting the dataset into the Training set and Test set
#데이터 세트를 훈련 세트와 테스트 세트로 나눈 후 특성 스케일링을 적용해야 한다.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1) #권장 훈련 세트 80%, 테스트 세트 20%
# In[ ]:
# Feature Scaling -> 하는 이유 => 특성 스케일이 다르면 머신러닝 알고리즘이 잘 작동하지 않는다.
#표준화 = -3 ~ 3 사이의 값 -> 항상 좋은 값이 나온다.
#정규화 = 대부분의 특성이 정규 분포를 따른다는 특수한 상황에서만 좋다.

#테스트 세트는 fit()할 필요가 없다. -> transfrom만 한다.
#특성 행렬의 가변수에도 특성 스케일링의 표준화를 적용해야 하나? -> 안 한다.
#표준화 = 모든 특성의 값을 동일한 범위로 변환하는 것 -> 가변수는 0, 1 이미 -3 ~ 3 사이에 있다. => 의미가 없다.
#가변수의 의미가 사라진다.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, [3,4]] = sc.fit_transform(X_train[:, [3,4]])
X_test[:, [3,4]] = sc.transform(X_test[:, [3,4]])
#fit()은 각 특성의 평균과 표준편차를 가져올 뿐
#transform()은 이 식을 적용해 모든 값의 스케일을 맞춘다.

print(X_train)
print()
print(X_test)

