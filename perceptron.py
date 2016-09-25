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
        self.errors = []

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


