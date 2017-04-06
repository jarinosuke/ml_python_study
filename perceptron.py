# coding:utf-8
import numpy as np

class Perceptron(object):
    """ パーセプトロンの分類器

    param
    ---

    eta : float
    学習率(0.0~1.0)

    n_iter : int
    トレーニングデータを何回トレーニングするかの回数

    attributes
    ---

    w_ : 1次元配列
    適合後の重み

    errors_ : リスト
    各エポックでの誤分類の数
    エポック=トレーニングの試行indexみたいなもの？

    """

    def __init__(self, eta=0.01, n_iter=10) :
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y) :
        """ トレーニングデータに適合させる

        param
        ---

        X : 配列のようなデータ構造 shape = [n_samples, n_features] = サンプル数x特徴数
        y : shape = n_samples = サンプル数

        return value
        ---
        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y) :
                # 重み w1...wm の更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                
                # 重み w0 の更新
                self.w_[0] += update

                # 重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)

            # for毎に誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X) :
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X) :
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

import matplotlib.pyplot as plt
import numpy as np

# 1-100行目の目的編集の抽出
y = df.iloc[0:100, 4].values

# Iris-setosaを-1、Iris-virginicaを1にする
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目のの1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values

# 品種 setosa のプロット
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

# 品種 virginica のプロット
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# 軸ラベルの設定
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')

plt.legend(loc='upper left')

plt.show()

# パーセプトロン作成
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# エポックとご分類誤差の関係折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # 活性化関数の計算
            output = self.net_input(X)

            # 誤差の計算
            errors = y - output

            # w の更新
            self.w_[1:] += self.eta * X.T.dot(errors)

            # w0 の更新
            self.w_[0] += self.eta * errors.sum()

            # cost 関数の計算
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    # 総入力の計算
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 線形活性化関数の出力を計算
    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
# plot_decision_regions(X_std, y, classifier=ada) not implemented this function
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standalized]')
plt.ylabel('petal length [standalized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
