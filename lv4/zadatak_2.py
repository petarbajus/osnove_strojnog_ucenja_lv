import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# učitaj podatke
data = pd.read_csv('lv4/data_C02_emission.csv')

# numeričke varijable
numerical_data = data.select_dtypes(include='number')
target = 'CO2 Emissions (g/km)'

X_num = numerical_data.drop(target, axis=1)

# kategorijska varijabla Fuel Type → one-hot encoding
X_cat = pd.get_dummies(data['Fuel Type'], prefix='Fuel')

# spoji numeričke i kategorijske
X = pd.concat([X_num, X_cat], axis=1)
y = data[target]

# split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# standardizacija svih varijabli
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# predikcija na test skupu
y_pred = model.predict(X_test_scaled)

# maksimalna apsolutna pogreška
errors = abs(y_test - y_pred)
max_error = errors.max()
max_error_index = errors.idxmax()  # indeks u test skupu

print(f"Maksimalna pogreška: {max_error:.2f} g/km")

# podatci o vozilu s maksimalnom pogreškom
vehicle_max_error = data.loc[max_error_index, ['Make','Model','Vehicle Class','CO2 Emissions (g/km)','Fuel Type']]
print("\nVozilo s maksimalnom pogreškom:")
print(vehicle_max_error)