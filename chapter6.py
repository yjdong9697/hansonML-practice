import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:] # 꽃잎의 길이와 너미
y = iris.target

# Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth = 2)
dt.fit(X, y)

# Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth = 2)
dtr.fit(X, y)
