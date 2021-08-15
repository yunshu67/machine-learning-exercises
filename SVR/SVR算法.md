#! https://zhuanlan.zhihu.com/p/400028363

# SVR 算法

# 1. SVR的背景

SVR全称是support vector regression，是SVM（支持向量机support vector machine）对回归问题的一种运用。在之前的部分中有提到过SVM的原理及其用法，这里就不再赘述了。这里为大家提供了一张图来直观的理解SVM和SVR的区别和联系：

![Image Name](https://cdn.kesci.com/upload/image/qvwyvs7r06.png?imageView2/0/w/960/h/960)

# 2. 模型原理

SVR模型可以简单理解为，在线性函数的两侧创造了一个“间隔带”，而这个“间隔带”的间距为ϵ（这个值常是根据经验而给定的），对所有落入到间隔带内的样本不计算损失，也就是只有支持向量才会对其函数模型产生影响，最后通过最小化总损失和最大化间隔来得出优化后的模型。对于非线性的模型，与SVM一样使用核函数（kernel function）映射到特征空间，然后再进行回归。
![Image Name](https://cdn.kesci.com/upload/image/qvwyzlh78h.png?imageView2/0/w/960/h/960)

上图显示了SVR的基本情况：

1. f(x)=wx+b是我们最终要求得的模型函数
2. wx+b+ϵwx+b-ϵ(也就是f(x)+ ϵ和f(x)- ϵ)是隔离带的上下边缘
3. ξ∗是隔离带下边缘之下样本点，到隔离带下边缘上的投影，与该样本点y值的差。

## 公式表述：

![Image Name](https://cdn.kesci.com/upload/image/qvwz4b5vb3.png?imageView2/0/w/960/h/960)
SVR模型在使用中和传统的一般线性回归模型也有一些的区别，其区别主要体现在：

1. SVR模型中当且仅当和f(x)和y之间的差距的绝对值大于ϵ时才计算损失，而一般的线性模型中只要f(x)和y不相等就计算损失。
2. 两种模型的优化方法不同，SVR模型中通过最大化间隔带的宽度和最小化损失来优化模型，而在一般的线性回归模型中是通过梯度下降后的平均值来优化模型。

# SVR的应用

SVR算法可以使用Scikit-Learn的SVR类来实现。应用方式如下：

```
from sklearn.svm import SVR  
regressor = SVR(kernel = 'rbf') #参数kernel是用来指定使用的核函数  
regressor.fit(x,y)
```

下面给出一个具体的应用:
数据中给出了一个薪资和职位的关系

![Image Name](https://cdn.kesci.com/upload/image/qvwzc42y48.png?imageView2/0/w/960/h/960)

## 1. 导入数据

In [12]:

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/home/mw/input/svrdemo9201/Demo data.CSV')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
```

## 2. 对数据进行处理，并对数据进行标准化

In [13]:

```
X = np.reshape(X, (-1, 1))
Y = np.reshape(Y, (-1, 1))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
```

## 3. 使用SVR模型对数据进行拟合

In [15]:

```
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)
```

## 4. 使用训练的SVR模型进行预测，并将得到的预测值转化为正常值

In [18]:

```
Y_pred = regressor.predict(sc_X.transform(np.array([[5.5]])))
Y_pred = sc_Y.inverse_transform(Y_pred)
```

## 5.对数据进行可视化处理

In [19]:

```
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

# 练习题

使用sklearn提供的SVR的API对波士顿房价数据集的数据进行预测，并尝试将预测的结果进行分析。波士顿房价的数据集使用方法如下:

```
from sklearn.datasets import load_boston  
boston = load_boston()
```

# 参考链接

1. [https://blog.csdn.net/weixin_41940690/article/details/106639347?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-6.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-6.control](https://blog.csdn.net/weixin_41940690/article/details/106639347?utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-6.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-6.control)
2. https://blog.csdn.net/qq_41909317/article/details/88542892

