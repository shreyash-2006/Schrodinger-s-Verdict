from sklearn.naive_bayes import GaussianNB
from data import x_train, y_train, x_valid, y_valid
from sklearn.metrics import classification_report


nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_valid)
print("Classification Report:\n", classification_report(y_valid, y_pred))
