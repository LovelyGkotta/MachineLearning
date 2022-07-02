
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
X = iris.data[:, [2, 3]]
y = iris.target  # 取species列，类别
print('Class labels:', np.unique(y))
# Output:Class labels: [0 1 2]
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

X_train.shape
# Output:(105, 2)
X_test.shape
# Output:(45, 2)
X.shape
# Output:(150, 2)
y_train.shape
# Output: (105,)
y_test.shape
# Output: (45,)

# scaler = sklearn.preprocessing.StandardScaler().fit(train)
# scaler.transform(train);scaler.transform(test)
# fit()方法建模，transform()方法转换
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # 初始化一个对象sc去对数据集作变换
sc.fit(X_train)  # 用对象去拟合数据集X_train，并且存下来拟合参数
# Output:StandardScaler(copy=True, with_mean=True, with_std=True)
# type(sc.fit(X_train))
# Output:sklearn.preprocessing.data.StandardScaler
sc.scale_  # sc.std_同样输出结果
# Output:array([ 1.79595918,  0.77769705])
sc.mean_
# Output:array([ 3.82857143,  1.22666667])

import numpy as np

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# test标准化原理
at = X_train_std[:5] * sc.scale_ + sc.mean_
a = X_train[:5]
at == a
# Output:
# array([[ True,  True],
#       [ True,  True],
#       [ True,  True],
#       [ True,  True],
#       [ True,  True]], dtype=bool)
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):  # Numpy版本检测函数
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 画决策边界,X是特征，y是标签，classifier是分类器，test_idx是测试集序号
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 第一个特征取值范围作为横轴
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 第二个特征取值范围作为纵轴
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # reolution是网格剖分粒度，xx1和xx2数组维度一样
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # classifier指定分类器，ravel是数组展平；Z的作用是对组合的二种特征进行预测
    Z = Z.reshape(xx1.shape)  # Z是列向量
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # contourf(x,y,z)其中x和y为两个等长一维数组，z为二维数组，指定每一对xy所对应的z值。
    # 对等高线间的区域进行填充（使用不同的颜色）
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)  # 全数据集，不同类别样本点的特征作为坐标(x,y)，用不同颜色画散点图

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]  # X_test取测试集样本两列特征，y_test取测试集标签

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')  # c设置颜色，测试集不同类别的实例点画图不区别颜色

from sklearn.linear_model import Perceptron
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
#ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn = Perceptron()  #y=w.x+b
ppn.fit(X_train_std, y_train)
#Output:Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
#      n_iter=5, n_jobs=1, penalty=None, random_state=0, shuffle=True,
#      verbose=0, warm_start=False)
ppn.coef_  #分类决策函数中的特征系数w
#Output:array([[-1.48746619, -1.1229737 ],
#       [ 3.0624304 , -2.18594118],
#       [ 2.9272062 ,  2.64027405]])
ppn.intercept_  #分类决策函数中的偏置项b
#Output:array([-1.,  0., -2.])
y_pred = ppn.predict(X_test_std)  #对测试集做类别预测
y_pred
#Output:array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0,
#       0, 2, 0, 0, 1, 0, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 1, 2, 0, 2, 0, 0])
y_test
#Output:array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0,
#       0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0, 0])
y_pred == y_test
#Output:array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True, False,  True,  True,  True,  True,  True,  True,  True,
#        True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True, False,  True,  True,  True,  True,  True,  True,  True,
#        True, False,  True,  True,  True,  True,  True,  True,  True], dtype=bool)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
#Output:Misclassified samples: 3
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  #预测准确度,(len(y_test)-3)/len(y_test):0.9333333333333333
#Output:Accuracy: 0.93
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()
