import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Data load
mnist = fetch_openml('mnist_784', version = 1)
X, y = mnist["data"], mnist['target']
y = y.astype(np.uint8)
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print(x_train.shape)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# 이진 데이터 훈련
sgd = SGDClassifier(random_state = 42)
sgd.fit(x_train, y_train_5)

# 성능 측정
y_train_pred = cross_val_predict(sgd, x_train, y_train_5, cv = 3)

print(precision_score(y_train_pred, y_train_5))
print(recall_score(y_train_pred, y_train_5))
print(f1_score(y_train_pred, y_train_5))

# 결정함수 값 저장
y_scores = cross_val_predict(sgd, x_train, y_train_5, cv = 3, method = "decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, threshold = precision_recall_curve(y_train_5, y_scores)

# Roc curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
## 면적 계싼
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5, y_scores))

# 다중 회귀(Multiple regression)
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(x_train, y_train)

