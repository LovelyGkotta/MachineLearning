import matplotlib.pyplot as plt
import random
import numpy as np

np.random.seed(2)

x = np.random.rand(50)  #x,yの乱数を作る
y =2*x + np.random.rand(50)

m = 0   #傾き
c = 0   #切片
SS_tot = 0
SS_res = 0
R_2 = 0
n=len(x)

L = 0.05    #学習効率
epochs = 1000   #学習回数

for i in range(epochs):
    y_pred= m*x+c
    D_m=(-2/n)*sum(x*(y-y_pred))
    D_c=(-2/n)*sum(y-y_pred)
    m=m-L*D_m
    c=c-L*D_c

y_ave = np.mean(y)

for j in range(n):
    SS_tot += (y[j]-y_ave)**2.0
    SS_res += (y[j]-(m*x[j]+c))**2.0

R_2 = 1.0-(SS_res/SS_tot)

print("傾き", "%.3f" % m)
print('切片', "%.3f" % c)
print("R_2", "%.3f" % R_2)

# plt.scatter(x, y,label="original data")
# plt.plot(x,y_pred,color="r")
# plt.legend()
# plt.show()
#
