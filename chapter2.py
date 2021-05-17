import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing,tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

## 테스트 세트 분리해놓는 작업(순수한 무작위 샘플링) 단, 샘플링 편향이 생길 가능성이 높음
housing["income_cat"] = pd.cut(housing['median_income'], bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits= 1, test_size = 0.2, random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

## 추가한 income_cat 특성을 각각 지움
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

## 훈련 세트에 대해서만 이용할 수 있도록 처리(단, 원본은 유지하기 위해 복사)
housing = strat_train_set.copy()

## 위도와 경도를 기준으로 데이터 분포를 그림 + 인구수가 많을수록 원이 더 큼 + 주택 가격을 색으로 표시
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1, s = housing["population"]/100,
            label = "population", figsize = (10, 7), c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)

plt.show()