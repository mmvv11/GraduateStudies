"""
데이터 전처리
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = np.column_stack((fish_length, fish_weight)) # concat과 같은 기능 ㅇㅇ 열방향 merge
fish_target = np.concatenate((np.ones(35), np.zeros(14))) # 1을 35개 0을 14개 ㅇㅇ

# 사이킷런이 알아서 샘플링 해줌 train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

# 요상한 도미 한 마리
# 길이 25, 무게 150이면 모델에 의하면 도미
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
print("1을 기대했지만, 0을 반환")
print(kn.predict([[25, 150]]))


"""
시각화된 내용을 확인해보아도 도미 데이터와 가깝지만, 빙어라고 return하는 이유는??
n_neighbors 기본값이 5로 설정되어 최근접 5개에 대한 값을 return 하는 것이기 때문임.
"""
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

"""
주어진 샘플에서 가장 가까운 이웃을 찾아주는 메서드, kneighbors
"""
distances, indexes = kn.kneighbors([[25, 150]])
print(f"distances: {distances},\ntrain_target: {train_target[indexes]}")

plt.title("스케일 조정하기 전")
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

plt.title("스케일 조정한 후")
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

"""
데이터 표준화하기

표준점수(Z) = (x - mean) / std
"""

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std

plt.title("표준화 된 그래프")
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 샘플도 표준화 ㄱㄱ
new = ([25, 150] - mean) / std

plt.title("표준화 된 그래프에서 샘플의 위치 시각화")
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

"""
표준화 된 데이터로 모델 학습
"""
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
# 모델 결과 확인 ==> 1 return
print(f"표준화 된 데이터 학습 후 모델 결과: {kn.predict([new])}")

distances, indexes = kn.kneighbors([new])

plt.title("표준화 최종 그래프")
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()