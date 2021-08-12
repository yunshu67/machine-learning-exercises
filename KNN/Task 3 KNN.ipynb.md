#! https://zhuanlan.zhihu.com/p/399204247
我们知道，鸢尾花分为不同的种类，但种类是如何被划分的呢？同一种类的鸢尾花具有哪些公共特征？某个类型的鸢尾花的某些特征是否更加频繁？基于此，本章将介绍k-近邻算法，它十分的有效且易于掌握。通过k-近邻法构建程序，我们可以自动划分鸢尾花的类型。接下来，我们将通过探讨k-近邻算法的基本理论以及实际例子进行讲解。
# K-近邻算法基本理论
### 算法概述
k-近邻算法就是采用测量不同特征值之间的距离进行分类的方法。它的思路是：如果一个样本在特征空间中的k个最相似（邻近）的样本中大多数属于一个类别，则该样本也属于这个类别。在K-近邻算法当中，所选择的邻近点都已经是正确分类的对象。我们只依据k个（通常不大于20）邻近样本的类别来决定待分样本的类别。
### 算法流程
k-近邻算法的一般流程是：
        1. 收集数据
        2. 计算待测数据与训练数据之间的距离（一般采用欧式距离）
        3. 将计算的距离排序
        4. 找出距离最小的k个值
        5. 计算找出值中每个类别的频次
        6. 返回最高频次的类别
### 算法特点
优点：精度高、对异常值不敏感
缺点：计算复杂度高、空间复杂度高
# 如何使用代码实现数据导入并分析数据
以鸢尾花数据集为例，鸢尾花数据集内包括3类鸢尾，包括山鸢尾、变色鸢尾和维吉尼亚鸢尾，每个记录都有4个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度。
## 数据集导入与分析


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
```


```python
# 加载数据集
dataset = load_iris()
# 划分数据
X_train, X_test, y_train, y_test = train_test_split(dataset['data'],dataset['target'],random_state= 0)
# random_state的作用相当于随机种子，是为了保证每次分割的训练集和测试集都是一样的
# 设置邻居数,即n_neighbors的大小
knn = KNeighborsClassifier(n_neighbors = 5)
# 构建模型
knn.fit(X_train,y_train)
# 得出分数
print("score:{:.2f}".format(knn.score(X_test,y_test)))
```


```python
# 我们也可以单独对某一数据进行测试
# 尝试一条测试数据
X_try = np.array([[5,4,1,0.7]])
# 对X_try预测结果
prediction = knn.predict(X_try)
print("prediction = ",prediction)

```

得出结果：
prediction =  [0]
即这朵花是山鸢尾

# 作业

使用sklearn提供的KNN的API对手写数字数据集的数据进行预测，手写数字数据集使用方法如下:

```python
from sklearn.datasets import load_digits
digits = load_digits()
```
# 参考来源

1. 《机器学习实战》Peter Harrington
2. 《统计学习方法》李航
3. 《Python机器学习基础教程》


```python

```
