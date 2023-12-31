""""
컨볼루션 계층 = 합성곱, 행렬 곱셈의 원리를 비롯한 선형 대수학을 활용해 이미지 내의 패턴을 식별한다.
입력과 필터로부터의 연속된 점곱으로 이루어진 최종 출력을 기능 맵, 활성화 맵 또는 컨볼빙된 기능
필터 크기는 일반적으로 3X3 행렬
특성 탐지기 -> 특성 맵을 얻을 수 있다. 커널, 필터 사용
"""

"""
ReLU 게층
    각 컨볼류션 작업 후 CNN은 기능 맴에 ReLU(Rectified Linear Unit) 변환을 적용하여 모델에 비선형성을 도입
"""

"""
풀링 계층 = 다운샘플링
    차원 감소를 수행하여 입력의 매개변수 수를 줄인다.
    풀링 작업은 전체 입력에 걸쳐 필터를 스윕하지만 이 필터에는 가중치가 없다. -> 수용 필드 내의 값에 집계 함수를 적용해 출력 배열을 채운다.
    최대 풀링 = 필터가 입력을 가로질러 이동하면서 최대 값을 가진 픽셀을 선택해 출력 배열로 보낸다. => 많이 사용
    합계 풀링 = 
    풀링 계층에서 많은 정보가 손실되지만 이점 / 복잡성을 줄이고 효율성을 개선하며 과적합의 위험을 제한
"""

"""
Flattening
열로 만든다.
얻은 출력값의 벡터 또는 전체 열을 입력 층에 넣어준다.
"""

"""
완전 연결 계층(FC)
    부분 연결 계층에서는 입력 이미지의 픽셀 값이 출력 계층에 직접 연결되지 않는다.
    출력 계층의 각 노드가 이전 계층의노드에 직접 연결
    이전 계층과 다른 필터를 통해 추출된 기능을 기반으로 분류 작업을 수행
    컨볼류션 계층과 풀링 계층은 ReLU 함수를 사용하는 경향
    FC 계층은 일반적으로 소프트맥스 활성화 함수를 활용하여 입력값을 적절하게 분류해 0~1까지의 확률을 생성
    
    컨볼루션 계층 -> 풀링 계층 -> 플래트닝 -> 완전 연결 계층
"""

"""
소프트맥스
    세 개 이상으로 분류하는 다중 클래스 분류에서 사용되는 활성화 함수
    분류될 클래스가 n개이면 n차원의 벡터를 입력받아 각 클래스에 속할 확률 추정
    확률의 총ㄹ합 = 1 / 어떤 분류에 속할 확률이 가장 높은지 쉽게 인지할 수 있다.
교차 앤트로피
    앤트로피 = 변수의 불확실성을 나타내는 지표 / 하나의 변수가 가지는 확률 분포의 불확실성을 의미
    하나의 변수가 지는 확률 분포의 불확실성을 의미
    정답 y가 {0, 1}에 속할 때 가지는 P는 정답, Q는 예측 출력
    잘못된 손실은 무한대로, 잘된 손실은 0값
    
평균제곱오차 MSE
    작을 수록 알고림의 성능이 좋다
    (정답 - 예측)^2 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator



