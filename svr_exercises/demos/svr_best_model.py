from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Carrega o dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline com escalonamento + SVR
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])

# Hiperparâmetros para cada kernel
param_grid = [
    {
        "svr__kernel": ["linear"],
        "svr__C": [0.1, 1, 10],
        "svr__epsilon": [0.1, 0.2, 0.5]
    },
    {
        "svr__kernel": ["rbf"],
        "svr__C": [1, 10],
        "svr__epsilon": [0.1, 0.2],
        "svr__gamma": ["scale", 0.01, 0.1]
    },
    {
        "svr__kernel": ["poly"],
        "svr__C": [0.1, 1],
        "svr__epsilon": [0.1],
        "svr__degree": [2, 3],
        "svr__gamma": ["scale"]
    }
]

# Busca com validação cruzada
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Avaliação no conjunto de teste
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Resultados
print(f"Melhor combinação de parâmetros: {grid_search.best_params_}")
print(f"MSE no conjunto de teste: {mse:.4f}")
