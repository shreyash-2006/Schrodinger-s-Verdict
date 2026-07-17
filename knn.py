from sklearn.neighbors import KNeighborsClassifier
from data import x_train, y_train, x_valid, y_valid
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_valid)

print("Classification Report:\n", classification_report(y_valid, y_pred))
