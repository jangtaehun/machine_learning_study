from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape) #60000개 이미지, 28X28 크기
print(train_target.shape) #60000개의 원소

#fig = 데이터가 담기는 프레임 = 액자
#axs = 데이터가 그려지는 캔버스 = 내부 좌표축
#subplot = 여러 개의 그래프를 배열처럼 쌓을 수 있게 도와준다. ex) plt.subplot(3, 2) = 3행 2열
fig, axs = plt.subplots(1, 10, figsize=(10, 10)) # 1행 10열 갠버스에 이미지 출력
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))

from sklearn.preprocessing import StandardScaler
train_input = train_input.reshape(-1, 28*28)
sc = StandardScaler()
train_scaled = sc.fit_transform(train_input)

"""
#표준화를 사용 안 했을 때 
train_scaled = train_input / 255.0
#각 픽셀은 0~255 사이의 정수값 -> 255로 나누어 0~1 사이 값으로 정규화
train_scaled = train_scaled.reshape(-1, 28*28)
#reshape(-1) -> 자동으로 남은 차원 할당, 두 번째 세 번째 차원을 1차원으로 합친다.
#784개의 픽셀로 이루어진 60000개의 샘플
#SDG는 2차원 입력을 다루지 못 해서 1차원 배열로 만든다.
"""

#로지스틱 회귀로 분류
#가장 기본적인 인공신경망은 확률적 경사하강법을 사용하는 로지스틱 회귀
from sklearn.model_selection import cross_validate #cross_validate 데이터 교차 검증
from sklearn.linear_model import SGDClassifier
sg = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sg, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
#'test_score' = 교차검증의 최종 점수
#'fit_time' = 모델 훈련 소요시간
#'score_time' = 모델 검증 소요시간

import tensorflow as tf
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
#출력층 = 10개 뉴런
#이진분류 - 시그모이드, 다중분류 - 소프트맥스 -> 활성화 함수
#입력 크기

model = keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=10)
#이진분류 = 출력층 뉴런 1개 / 양성 클래스에 대한 확률만 출력 => 음성 클래스 = 1 - a
#다중분류 = 타깃에 해당하는 확률만 남겨 놓기 위해 나머지 확률에 0을 곱한다.
#텐서플로는 정수로 된 타킷값을 원핫인코딩을 안 해도 된다. -> 원핫인코딩(O) = categorical_crossentropy
model.evaluate(val_scaled, val_target)
#케라스에서 모델 성능확인 메서드 =  evaluate()

"""
사이킷런
    모델 - sc = SGDclassifier(loss='log_loss', max_iter=5)
    훈련 - sc.fit(train_scaled, train_target)
    평가 - sc.scores(val_scaled, val_target)
    
케라스
    dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    모델 - model = keras.Sequential(dense)
    model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    훈련 - model.fit(train_scaled, train_target, epochs=5)
    평가 - model.evaluate(val_scaled, val_target)
"""