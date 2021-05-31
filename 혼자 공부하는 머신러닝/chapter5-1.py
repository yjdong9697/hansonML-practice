import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# Logistic regression (Underfitting)
lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print("Logistic Regresssion result")
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print()

# Decision Tree
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)

print("Decision Tree result")
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
print()

# Visualize Decision Tree result
plt.figure(figsize = (10, 7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()