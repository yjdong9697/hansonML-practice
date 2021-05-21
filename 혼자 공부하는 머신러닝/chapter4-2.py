import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

fish = pd.read_csv("https://bit.ly/fish_csv")

fish_input = fish[['Weight', 'Length', 'Diagonal', "Height", 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# Standardize
ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss = 'log', max_iter = 10, random_state = 42)
sc.fit(train_scaled, train_target)

print("Stochastic gradient descent result")

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
print()

sc = SGDClassifier(loss = 'log', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show() # epoch가 100정도가 적당함을 파악할 수 있음

sc = SGDClassifier(max_iter = 100, random_state = 42, loss = 'log')
sc.fit(train_scaled, train_target)
print("Epoch를 100으로 조정한 Stochastic gradient descent method result")
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))