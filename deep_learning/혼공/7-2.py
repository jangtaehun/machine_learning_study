from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape) #60000개 이미지, 28X28 크기
print(train_target.shape) #60000개의 원소

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
train_input = train_input.reshape(-1, 28*28)
sc = StandardScaler()
train_scaled = sc.fit_transform(train_input)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

"""
입력층 - 출력층 사이에 은닉층
    은닉층 = 활성화 함수 사용 -> 신경망 층의 선형 방정식의 계산 값에 적용하는 함수
    
    출력층
        이진 분류 => 시그모이드 함수
        다중 분류 => 소프트맥스 함수
    은닉층
        활성화 함수 선택이 비교적 자유롭다.
        초반 = 시그모이드 함수 많이 사용
            오른쪽과 왼쪽 끝으로 갈 수록 그래프가 누워있어서 올바른 출력을 만드는데 신속하게 대응하지 못 한다.
            층이 많은 심층 신경망일 수록 학습을 어렵게 한다.
        최근 = 렐루 함수 사용
            입력이 양수일 경우 활성화 함수가 없는 것처럼 입력을 통과 시키고 음수일 경우 0으로 만든다.
            이미지 처리에서 좋은 성능
            
        은닉층의 뉴런 개수 = 특별한 기준 없다. 출력층의 뉴런보다 많게
        은닉층에서 성형적인 산술 계산만 수행하면 수행 역할이 없다. 따라서 선형 계산을 비선형적으로 비틀어 주어야 한다.
        => 다음 층의 계산과 단순히 합쳐지지 않고 나름의 역할을 한다.
        a * 4 + 2 = b // b * 3 - 5 = c => a * 12 + 1 = c
        
        a * 4 + 2 = b // log(b) = k // b * 3 - 5 = c => k * 3 - 5 = c
"""

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='relu', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.summary()

"""
인공신경망에 주입 => 1차원
    Flatten = 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼친다.
"""

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_input = train_input.reshape(-1, 28*28)
sc = StandardScaler()
train_scaled = sc.fit_transform(train_input)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28))) #위에서 reshape를 해서 생략 가능
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

"""
keras = 기본적으로 미니배치 경사 하강법
    미니배치
        개수 = 기본 32
        fit() -> batch_size로 조정 가능
        
compile()
    경사 하강법 선택 -> 옵티마이저
    keras 기본 경사 하강법 = RMSprop -> 옵티마이저
    다양한 옵티마이저 제공
        SGD(확률적 경사 하강법) = 가장 기본적인 옵티마이저 : optimizer = 'sgd'
            momentum: 기본값 0 -> 0보다 큰 값 지정해 모멘텀 최적화 사용 / 보통 0.9 이상 지정
            nesterov: 기본값 False -> True = 네스테로프 모멘텀 최적화 / 모멘텀 최적화 2번 반복해 구현
        적응적 학습률 = 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있다. -> 안정적으로 최적점에 수렴할 가능성이 높다. / learning_rate=0.001
             Adagrad
             RMSprop 
             Adam = 모멘텀 최적화 + RMSprop 장점 접목
"""

model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)