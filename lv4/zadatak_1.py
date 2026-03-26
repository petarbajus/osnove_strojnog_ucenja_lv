import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

path = r"C:\Users\student\Desktop\LV4_pb_osu\osnove_strojnog_ucenja_lv\lv4\data_C02_emission.csv"
data = pd.read_csv(path)


#----------------------------------------
# a)
#----------------------------------------
numerical_data = data.select_dtypes(include='number')

target = 'CO2 Emissions (g/km)'

X = numerical_data.drop(target, axis=1)
y = numerical_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Features:", X.columns.tolist())

#----------------------------------------
# b)
#----------------------------------------


features = X.columns
n = len(features)

cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(15, 5 * rows))

for i, feature in enumerate(features, 1):
    plt.subplot(rows, cols, i)
    
    plt.scatter(X_train[feature], y_train, color='blue', label='Train', alpha=0.6)
    plt.scatter(X_test[feature], y_test, color='red', label='Test', alpha=0.6)
    
    plt.xlabel(feature)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.title(f'{feature} vs CO2')
    plt.legend()

plt.tight_layout()

#----------------------------------------
# c)
#----------------------------------------


scaler = StandardScaler()

# fit samo na train
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X_train.columns
n = len(feature_names)

cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(15, 5 * rows))

for i, feature in enumerate(feature_names, 1):
    idx = list(feature_names).index(feature)
    
    plt.subplot(rows, cols, i)
    
    # prije skaliranja
    plt.hist(X_train[feature], bins=30, alpha=0.5, label='Before', color='blue')
    
    # nakon skaliranja
    plt.hist(X_train_scaled[:, idx], bins=30, alpha=0.5, label='After', color='red')
    
    plt.title(feature)
    plt.legend()

plt.tight_layout()

#----------------------------------------
# d)
#----------------------------------------


# model
model = LinearRegression()

model.fit(X_train_scaled, y_train)

print("Intercept (w0):", model.intercept_)
print("Koeficijenti (w1 ... wn):")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"{feature}: {coef}")
    
#----------------------------------------
# e)
#----------------------------------------

# predikcija CO2 emisija na test skupu
y_pred = model.predict(X_test_scaled)

# scatter plot: stvarno vs predviđeno
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')

# dijagonala y=x za referencu
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)

plt.xlabel("Stvarne vrijednosti CO2 (g/km)")
plt.ylabel("Predviđene vrijednosti CO2 (g/km)")
plt.title("Stvarno vs Predviđeno - Test skup")
plt.grid(True)

#----------------------------------------
# f)
#----------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

r2 = r2_score(y_test, y_pred)

print("Evaluacija modela na test skupu:")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

#----------------------------------------
# g)
#----------------------------------------

feature_names = X_train.columns

mse_list = []

for i in range(1, len(feature_names) + 1):
    selected_features = feature_names[:i]
    
    scaler = StandardScaler()
    X_train_subset_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_subset_scaled = scaler.transform(X_test[selected_features])
    
    model = LinearRegression()
    model.fit(X_train_subset_scaled, y_train)
    
    y_pred = model.predict(X_test_subset_scaled)
    mse = mean_squared_error(y_test, y_pred)
    
    mse_list.append(mse)

plt.figure(figsize=(8,6))
plt.plot(range(1, len(feature_names) + 1), mse_list, marker='o')
plt.xticks(range(1, len(feature_names) + 1), feature_names, rotation=45)
plt.xlabel("Broj varijabli")
plt.ylabel("MSE na test skupu")
plt.title("Utjecaj broja ulaznih varijabli na MSE")
plt.grid(True)
plt.show()