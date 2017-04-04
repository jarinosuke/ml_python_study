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
