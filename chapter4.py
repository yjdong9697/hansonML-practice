import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression

# Logistic regression
iris = sklearn.datasets.load_iris()
X = iris["data"][:, 3:] # 꽃잎의 너비만 가져옴
y = (iris["target"] == 2).astype(np.int) # 1 Iris-Virginica면 1로 저장(Broadcast)

log = LogisticRegression()
log.fit(X, y)
