from sklearn.datasets import make_regression
from LinearRegression import RunLinearRegression

X, Y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, noise=0.5, random_state=1)

# independent variables == n_informative
# dependent variables == n_targets

print(X.shape, Y.shape)
x = RunLinearRegression(X, Y)
x()
