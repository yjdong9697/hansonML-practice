import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Learn data
df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()

# Target data
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# Splitting train set and test set
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)

# Multiple regression by using feature engineering (여러 개의 특성을 활용하여 다중 회귀)
# 특성을 만드는 과정에서 특성을 사용해 새로운 특성을 만들어 내는 특성 공학(Feature engineering)을 진행하게 됨
poly = PolynomialFeatures(include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# Linear regression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print("1차항까지만 고려한 다중 회귀의 피팅 결과 출력")
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
print()
# Multiple regression using higher-order features
poly_deep = PolynomialFeatures(include_bias = False, degree = 5) # 5제곱까지 고려해서 다중 회귀
poly_deep.fit(train_input)
train_poly_deep = poly_deep.transform(train_input)
test_poly_deep = poly_deep.transform(test_input)

lr_deep = LinearRegression()
lr_deep.fit(train_poly_deep, train_target)
print("5차항까지만 고려한 다중 회귀의 피팅 결과 출력")
print(lr_deep.score(train_poly_deep, train_target))
print(lr_deep.score(test_poly_deep, test_target)) # -144.4057 (Overfitting)
print()

# Regulation
ss = StandardScaler()
ss.fit(train_poly_deep)
train_scaled = ss.transform(train_poly_deep)
test_scaled = ss.transform(test_poly_deep)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("5차항 까지 고려한 리지 회귀의 피팅 결과 출력")
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
print()

# Regulatory strength control by using R^2 (Determine alpha) : Ridge regression
# 결과를 보면 왼쪽은 과대적합, 오른쪽은 과소적합의 경향성을 띄고 있는 것을 쉽게 파악할 수 있다.
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()

# Regulatory strength control by using R^2 (Determine alpha) : Lasso regression
# 결과를 보면 왼쪽은 과대적합, 오른쪽은 과소적합의 경향성을 띄고 있는 것을 쉽게 파악할 수 있다.
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    lasso = Lasso(alpha = alpha)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()

# 결과를 보면, alpha가 10일때 가장 최적의 상태임을 쉽게 파악할 수 있다.
lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print("5차항 까지 고려한 리지 회귀의 피팅 결과 출력(최적의 alpha를 설정한 결과)")
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))





