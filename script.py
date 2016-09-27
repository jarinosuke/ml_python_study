import pandas as pd

# 読み込んだCSVの最後の5行を表示
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()


import matplotlib.pyplot as plt
import numpy as np

# データ整形

# 1-100行の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1、Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行の1,3列の抽出
X = df.iloc[0:100, [0, 2]].values

# setosaのプロット
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
# versicolorのプロット
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
# 軸の設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')
# 図の表示
# plt.show()


from perceptron import *
# パーセプトロンのオブジェクトの生成
ppn = Perceptron(eta=0.1, n_iter=10)
# トレーニングデータをモデルへ適合
ppn.fit(X, y)
# epocと誤分類誤差の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
# 図の表示
plt.show()

