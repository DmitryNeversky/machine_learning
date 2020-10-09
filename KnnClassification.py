from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt

X, y = make_moons(n_samples=1000, noise=0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

accuracy = accuracy_score(y_test, knn.predict(X_test))

print(accuracy)

x_space = np.linspace(-2, 2, 100)
x_grid, y_grid = np.meshgrid(x_space, x_space)
xy = np.stack([x_grid, y_grid], axis=2).reshape(-1, 2)
plt.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.1, c=knn.predict(xy))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()