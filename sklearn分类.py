import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import pandas as pd

df = pd.read_csv('C:\\Users\\32885\\Desktop\\meachinelearning\\Lib\\site-packages\\sklearn\\datasets\\data\\iris.csv')
print(df)

df.plot.scatter('petal.length','petal.width',c ='variety',colormap='jet')
plt.show()
x = df
ppn = Perceptron()
ppn.fit(petal.length,petal.width)

ppn.coef_
ppn.intercept_

y_pred = ppn.predict(x)

print("傾き",ppn.coef_)
print('切片',ppn.intercept_)
print("R_2",ppn.score(x,y))


# plot_decision_regions(x, y_pred, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
#
# plt.show()

