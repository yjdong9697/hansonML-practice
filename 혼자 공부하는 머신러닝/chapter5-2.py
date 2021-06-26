import numpy as np
from numpy.core.numeric import cross
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

dt = DecisionTreeClassifier(random_state = 42)

params = {'min_impurity_decrease' : [0.001, 0.002, 0.003, 0.004, 0.005]}
gs = GridSearchCV(dt, params, n_jobs = -1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_

print(dt.score(train_input, train_target))