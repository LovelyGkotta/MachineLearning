import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()
x = iris.data[:, 2:4]
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train ,marker ="x", cmap=plt.cm.gnuplot)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

ppn = LogisticRegression()
ppn.fit(x_train,y_train)
y_pred = ppn.predict(x)

print("傾き",ppn.coef_)
print('切片',ppn.intercept_)
print("R_2",ppn.score(x_train,y_train))

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.gnuplot)
plt.show()