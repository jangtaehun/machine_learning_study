# 활성화 함수
# 시그모이드 함수
# 하이퍼볼릭탄젠트 함수

#이진값에 사용할 수 있는 함수
#한계값 활성화 함수
#시그모이드 함수 -> y가 1일 확률을 알려준다

#정류화 활성화 함수

#입력 -> 활성화함수 - 정류화 활성화 함수 -> 출력(시그모이드함수 사용...)

#인공신경망은 일련의 층으로 구성되어 있다. 입력층부터 시작해서 최종 출력층까지 완전히 연결되어 있다. -> 일련의 층
#다른 유형으로 계산 그래프 -> 뉴런들이 연결되어 있지만 연속된 층으로 구성되어 있지는 않다. = 볼츠만 머신

#확률적 경사하강법 = 예측과 실제 결과 사이의 손실 오차를 줄이기 위해 가중치 업데이트

#데이터 불러오기 -> 결측치 -> 원핫 인코딩/레이블 인코딩 -> 세트 나누기 -> 표준화 -> ...

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv("./Churn_Modelling.csv")
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(keras.layers.Dense(10, activation="relu"))
ann.add(keras.layers.Dense(10, activation="relu"))
ann.add(keras.layers.Dropout(0.3))
ann.add(keras.layers.Dense(1, activation="sigmoid"))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

check_cb = keras.callbacks.ModelCheckpoint('test.h5', save_best_only=True)
early_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

ann.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[check_cb, early_cb],validation_data=(X_test, y_test))
ann.summary()

pred = ann.predict(X_test)
pred = (pred > 0.5)
print(pred)
print()
print(np.concatenate((pred.reshape(len(pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, pred)
print(cm)
#(1)이용 유지 1519명 예측
#(4)이용 취소 212명 예측
#(2)이용 취소 예측 76명 틀림
#(3)이용 유지 193명 틀림

finish = accuracy_score(y_test, pred)
print(finish)