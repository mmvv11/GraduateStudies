"""
훈련 세트와 테스트 세트

"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
               31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
               35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
               10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
               500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
               700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# length, weight를 zip으로 저장
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# 1이면 도미스, 0이면 빙어스
fish_target = [1] * 35 + [0] * 14

"""
트레인, 테스트 데이터 나누기
"""
train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]

"""
knn 모델 객체
"""
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
res = kn.score(test_input, test_target)
print(res)
print("============================")

"""
여기서는 res가 0임. 당연 => 도미만 가지고 훈련을 했음 => 샘플링도 잘해야한다 ㅇㅇ
넘파이로 한번 섞어주자.
"""
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42) # 난수 생성 초기값 설정
index = np.arange(49) # 49개 인덱스
np.random.shuffle(index) # 49개 인덱스틀 무작위로 섞어탱이 ㄱㄱ ㅋㅋ

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

"""
셔플된 데이터 시각화
"""
plt.scatter(train_input[:, 0], train_input[:, 1], color="red", label="train")
plt.scatter(test_input[:, 0], test_input[:, 1], color="blue", label="test")
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()

"""
셔플 데이터를 재학습 ㄱㄱ
"""
kn.fit(train_input, train_target)
res = kn.score(test_input, test_target)
predict = kn.predict(test_input)

print(f"suffle 후 res: {res},\npredict: {predict},\ntest_target: {test_target}")
