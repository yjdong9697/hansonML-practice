import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

## Data load by using keras
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

## Binary classifier 5 by using one-hot(Using SGD)
# Create bool array
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd = SGDClassifier(random_state = 42)
sgd.fit(x_train, y_train_5)

# 모델 성능 평가
print(cross_val_score(sgd, x_train, y_train_5, cv = 3, scoring = 'accuracy'))
