import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
train_input = train_input.reshape(-1, 28*28)
sc = StandardScaler()
train_scaled = sc.fit_transform(train_input)
train_scaled = train_scaled.reshape(-1, 28, 28)
#train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)


def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
#케라스 층을 추가하면 은닉층 뒤에 은닉층 추가

model = model_fn() # ex) model = model(keras.layers.Dropout(0.3))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
print(history.history.keys())
"""
verbose: 기본값 = 1
    1 -> 에포크마다 진행 막대, 손실 등의 막대(O)
    2 -> 진행 막대(X)
    0 -> 훈련 과정 표시(X)
    
history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어 있다.
"""

plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.show()

model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()

#검증 손실
"""
인공 신경망은 모두 일종의 경사 하강법 사용
에포크에 따른 과대, 과소적합 파악 -> 훈련세트/검증세트에 대한 점수 필요
정확도 -> 과대, 과소적합 설명(O)
손실을 통해 과대, 과소적합 설명 가능
    인공 신경망 모델이 최적화하는 대상 = 손실 함수
    
fit() 메서드에 검증 데이터 전달 => validation_data=(a, b)
"""
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
'''
초기에 검증 손실이 감소하다가 다섯 번째 에포크만에 다시 상승
훈련 손실은 꾸준히 감소 -> 전형적인 과대적합 모델
검증 손실이 상승하는 시점을 늦추면 세트에 대한 손실이 줄어들 뿐만 아니라 검증 세트에 대한 정확도 증가할 것
=> optimizer ='adam' : 적응적 학습률 -> 에포크가 진행되면서 학습률의 크기 조정
'''

#드롭 아웃
