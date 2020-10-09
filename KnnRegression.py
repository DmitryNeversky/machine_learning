from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

sv = GridSearchCV(KNeighborsRegressor(), param_grid={
    'n_neighbors': range(3, 12, 3),
    'weights': ['uniform', 'distance'],
    'p': [1,2,3]
}, cv=4)

sv.fit(X_train, y_train)

knn = KNeighborsRegressor(n_neighbors=sv.best_params_['n_neighbors'], weights=sv.best_params_['weights'], p=sv.best_params_['p'])

knn.fit(X_train, y_train)
mse = mean_squared_error(y_test, knn.predict(X_test))

print("Доля ошибок: ", int(mse), '%')

# metrics = []
#
# for n in range(2, 6):
#     knn = KNeighborsRegressor(n_neighbors=n)
#     knn.fit(X_train, y_train)
#     metrics.append(mean_squared_error(y_test, knn.predict(X_test)))
#
# plt.plot(range(2, 6), metrics)
# plt.ylabel(ylabel='Доля ошибок')
# plt.xlabel(xlabel='Кол-во соседей')
# plt.show()
