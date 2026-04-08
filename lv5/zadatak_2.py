import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv(r"C:\Users\student\Desktop\LV4_pb_osu\osnove_strojnog_ucenja_lv\lv5\penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# mapiranje stringova u brojeve
df['species'] = df['species'].map({
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
})

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------
# a)
#----------------------------------------

# y je oblika (n,1) → pretvori u 1D
y_train_1d = y_train.ravel()
y_test_1d = y_test.ravel()

# brojanje po klasama
classes_train, counts_train = np.unique(y_train_1d, return_counts=True)
classes_test, counts_test = np.unique(y_test_1d, return_counts=True)

# pozicije na x-osi
x = np.arange(len(classes_train))

plt.figure()
plt.bar(x - 0.2, counts_train, width=0.4, label='Train', tick_label=[labels[c] for c in classes_train])
plt.bar(x + 0.2, counts_test, width=0.4, label='Test')

plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasama (train vs test)')
plt.legend()
plt.show()

#----------------------------------------
# b)
#----------------------------------------

from sklearn.linear_model import LogisticRegression

# inicijalizacija modela za više klasa
model = LogisticRegression(solver='lbfgs', max_iter=500)
model.fit(X_train, y_train.ravel())

#----------------------------------------
# c)
#----------------------------------------

# intercepti (θ0) – po klasi
print("Intercepti (θ0) po klasi:\n", model.intercept_)

# koeficijenti (θ1, θ2) – po klasi
print("\nKoeficijenti (θ1, θ2) po klasi:\n", model.coef_)

#----------------------------------------
# d)
#----------------------------------------

# crtanje decision regions za train set
plot_decision_regions(X_train, y_train.ravel(), classifier=model)

plt.xlabel('Bill length (mm)')
plt.ylabel('Flipper length (mm)')
plt.title('Decision regions – Logistic Regression (train set)')
plt.show()

#----------------------------------------
# e)
#----------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("\nTočnost (accuracy):", accuracy)

report = classification_report(y_test, y_pred, target_names=[labels[c] for c in sorted(labels.keys())])
print("\nClassification report:")
print(report)

#----------------------------------------
# f)
#----------------------------------------

input_variables = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

X = df[input_variables].to_numpy()

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# treniranje logističke regresije
model = LogisticRegression(solver='lbfgs', max_iter=500)
model.fit(X_train, y_train.ravel())

# predikcija
y_pred = model.predict(X_test)

# evaluacija
accuracy = accuracy_score(y_test, y_pred)
print("Točnost (accuracy) sa 4 ulazne varijable:", accuracy)
print(classification_report(y_test, y_pred, target_names=[labels[c] for c in sorted(labels.keys())]))