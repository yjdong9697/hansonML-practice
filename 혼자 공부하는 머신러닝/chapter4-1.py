import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax

# Data preparation
fish = pd.read_csv("https://bit.ly/fish_csv")
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# Train set and test set separation and normalization
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# Probability guessing by using KNN
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)

proba = kn.predict_proba(test_scaled[0:5])
print("KNN을 활용한 확률 계산")
print(np.round(proba, decimals = 4))
print()

# Probability guessing by using logistic regression
# 도미와 빙어 데이터만 골라냄 by boolean indexing (이진 분류)
bream_smelt_indees = (train_target == "Bream") | (train_target == "Smelt")
train_bream_smelt = train_scaled[bream_smelt_indees]
target_bream_smelt = train_target[bream_smelt_indees]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

decision = lr.decision_function(train_bream_smelt[:5])
print("로지스틱 회귀를 통한 이진 분류 확률 예측")
print(expit(decision))
print()

# 다중 분류
lr = LogisticRegression(C = 20, max_iter = 1000) # C가 작을수록 규제가 커짐
lr.fit(train_input, train_target)

print("다중 회귀 결과 출력")
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print()

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3))

decision = lr.decision_function(test_scaled[:5])
proba = softmax(decision, axis = 1)
print(np.round(proba, decimals = 3))