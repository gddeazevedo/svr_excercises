import pandas as pd
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


X, y = make_regression(n_samples=10, n_features=1, noise=1, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

epsilon = 1
model = SVR(kernel='linear', epsilon=epsilon, C=1.0)
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

results = {
    "y": y,
    "f(x) = ŷ": y_pred,
    "ξ": [],
    "ξ'": [],
    "ε": epsilon,
    "Violação": [],
    "Violação acima": [],
    "Violação abaixo": []
}

for y_i, y_pred_i in zip(y, y_pred):
    xi      = max(0, y_i - y_pred_i - epsilon)
    xi_star = max(0, y_pred_i - y_i - epsilon)
    results["ξ"].append(xi)
    results["ξ'"].append(xi_star)
    results["Violação"].append(xi != 0 or xi_star != 0)
    results["Violação acima"].append(xi != 0)
    results["Violação abaixo"].append(xi_star != 0)

df = pd.DataFrame(results)
print(df)
