import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import linear_model

np.random.seed(2)

x = np.random.rand(50)
new_x = np.array(x).reshape(50,1)
y =2*x + np.random.rand(50)
new_y = np.array(y).reshape(50,1)

model = linear_model.LinearRegression()
model.fit(new_x,new_y)

y_pred = model.predict(new_x)
print("傾き", "%.3f" % model.coef_)
print('切片', "%.3f" % model.intercept_)
print("R_2", "%.3f" % model.score(new_x,new_y))

# plt.scatter(x, y,label="original data")
# plt.plot(x,y_pred,color="r")
# plt.legend()
# plt.show()
