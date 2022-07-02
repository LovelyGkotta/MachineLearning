import matplotlib.pyplot as plt
import random
import numpy as np

np.random.seed(0)

x = np.random.rand(5)  #x,yの乱数を作る
y =2*x + np.random.rand(5)

X_T=np.mat(x)
print(X_T)

X=X_T.T
print(X)
print("*****************")
cheng=np.matmul(X_T,X)
print(cheng)

ni=cheng.I
print(ni)

n1=np.matmul(cheng,X_T)
fin=np.matmul(n1,y)
print(fin)

# fin=ni*X_T*y
# print(fin)
# plt.scatter(x, y,label="original data")
# plt.legend()
# plt.show()

