import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#----------------------------------------
# a)
#----------------------------------------

plt.figure()

plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train,
    cmap='bwr',
    label='Train'
)

plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test,
    cmap='bwr',
    marker='x',
    label='Test'
)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prikaz podataka (train i test)')
plt.legend()

plt.show()

#----------------------------------------
# b)
#----------------------------------------

# inicijalizacija modela
model = LogisticRegression()

# učenje modela na train podacima
model.fit(X_train, y_train)

#----------------------------------------
# c)
#----------------------------------------

theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

print("theta0:", theta0)
print("theta1:", theta1)
print("theta2:", theta2)

# raspon x1 vrijednosti
x1_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

# odgovarajuće x2 vrijednosti (granica odluke)
x2_vals = -(theta0 + theta1 * x1_vals) / theta2

plt.figure()

# train podaci
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')

# granica odluke
plt.plot(x1_vals, x2_vals, color='black', label='Granica odluke')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke i train podaci')
plt.legend()

plt.show()

#----------------------------------------
# d)
#----------------------------------------
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# matrica zabune
cm = confusion_matrix(y_test, y_pred)

# metrike
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Matrica zabune:")
print(cm)

print("\nTočnost (accuracy):", accuracy)
print("Preciznost (precision):", precision)
print("Odziv (recall):", recall)

#----------------------------------------
# e)
#----------------------------------------

# logički indeksi
tocno = y_pred == y_test
netocno = y_pred != y_test

plt.figure()

# točno klasificirani (zeleno)
plt.scatter(
    X_test[tocno, 0],
    X_test[tocno, 1],
    color='green',
    label='Točno'
)

# pogrešno klasificirani (crno)
plt.scatter(
    X_test[netocno, 0],
    X_test[netocno, 1],
    color='black',
    label='Netočno'
)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Klasifikacija na testnom skupu')
plt.legend()

plt.show()
