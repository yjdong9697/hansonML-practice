import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Linear SVM
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # 꽃잎 길이, 꽃잎 너비
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C = 1, loss = 'hinge'))
])

svm_clf.fit(X, y)

print(svm_clf.predict([[5.5, 1.7]]))

# Non Linear SVM (There is a difference in wheter polynomial features is used or not)
X, y = datasets.make_moons(n_samples = 100, noise = 0.15)
polynomial_svm_clf = Pipeline([
    ('polynomial', PolynomialFeatures(degree = 3)),
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C = 10, loss = 'hinge'))
])

polynomial_svm_clf.fit(X, y)

# Kernel Trick (Not actually adding additional polynomial features)
poly_kernal_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("kernel", SVC(kernel = "poly", degree = 3, coef0= 1, C = 5))
])

poly_kernal_svm_clf.fit(X, y)

# Similatrity feature(Same as Kernel Trik, this model do not adding additional polynomial features)
# By using Gaussian RBF kernel
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf_kernel", SVC(kernel = "rbf", gamma = 5, C = 0.001))
])

rbf_kernel_svm_clf.fit(X, y)