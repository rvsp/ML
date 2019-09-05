from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

X_test = [[5.9, 1.0, 5.1, 1.8]]

y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
knn.score(X_test, y_test)
