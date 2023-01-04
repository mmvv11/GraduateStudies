from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(f"물고기 종류: {pd.unique(fish['Species'])}")

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']]
fish_target = fish['Species']
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

print(f"데이터 형태 train_input shape: {train_input.shape}")

# 데이터 정규화
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

"""
로지스틱 회귀 이진 분류 테스트

bream or smelt를 분류하는 모델
"""

# bream 또는 smelt를 가리키는 마스크
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]


lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(f"5개에 대한 예측 {lr.predict(train_bream_smelt[:5])}과\n확률 {lr.predict_proba(train_bream_smelt[:5])}")
print(f"모델이 가지고 있는 클래스 {lr.classes_}")
print(f"모델 파라미터 {lr.coef_, lr.intercept_}")

# 로지스틱 모델의 z값 => 이 값을 sigmoid에 대입하면 확률 값으로 return
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
# 시그모이드 값
print(f"sigmoid: {expit(decisions)}")
print(sum(expit(decisions)))

"""
로지스틱 다중 분류 테스트

"""
# 로지스틱은 기본적으로 반복적인 알고리즘을 사용한다.
# c는 릿지 회귀의 alpha 역할을 한다. 하지만 alpha와는 반대로 값이 커질수록 규제 강도가 낮아짐.
# c의 디폴트는 1, max_iter는 100
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(f"다중분류 predict {lr.predict(test_scaled[:5])}")

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
print(lr.classes_)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))