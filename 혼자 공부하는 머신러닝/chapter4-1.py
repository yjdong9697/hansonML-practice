import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
print(np.round(proba, decimals = 4))

# Probability guessing by using logistic regression
