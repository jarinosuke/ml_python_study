from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

print("Class labels:", np.unique(y))

# トレーニングデータとテストデータに分割
# 30% をテストデータに回す

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 標準化

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 平均と標準偏差
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: ', (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, y_pred))

