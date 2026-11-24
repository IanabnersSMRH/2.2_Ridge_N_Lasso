import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=200,
    n_features=5,
    n_informative=2,
    noise=10.0,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 1. LassoCV eligiendo alpha automáticamente
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)
preds_cv = lasso_cv.predict(X_test)

print("=== LassoCV ===")
print("Mejor alpha:", round(float(lasso_cv.alpha_), 4))
print("Coeficientes:", np.round(lasso_cv.coef_, 2))
print(
    f"MSE test: {mean_squared_error(y_test, preds_cv):.2f} | R2:"
    f" {r2_score(y_test, preds_cv):.3f}\n"
)

# 2. Reentrenar OLS usando solo las variables seleccionadas por Lasso (alpha=2.0)
lasso_sel = Lasso(alpha=2.0).fit(X_train, y_train)
mask = lasso_sel.coef_ != 0
indices_activos = np.where(mask)[0]

ols_limpio = LinearRegression()
ols_limpio.fit(X_train[:, mask], y_train)
preds_limpio = ols_limpio.predict(X_test[:, mask])

print("=== OLS solo con variables seleccionadas por Lasso ===")
print("Índices activos:", indices_activos)
print("Coeficientes:   ", np.round(ols_limpio.coef_, 2))
print(
    f"MSE test: {mean_squared_error(y_test, preds_limpio):.2f} | R2:"
    f" {r2_score(y_test, preds_limpio):.3f}"
)