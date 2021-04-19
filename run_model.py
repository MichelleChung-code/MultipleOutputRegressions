from sklearn.datasets import make_regression
from Regressions import RunRegression
import unittest
from unittest.mock import patch

X, Y = make_regression(n_samples=50, n_features=10, n_informative=5, n_targets=2, noise=0.5, random_state=1)

# independent variables == n_informative
# dependent variables == n_targets

print(X.shape, Y.shape)

print("=" * 30)
x = RunRegression(X, Y, 'LinearRegression')
x()


class TestRunModel(unittest.TestCase):

    # patch the plt.show() function since we don't need to see all the generated plots in the unittest
    @patch("Regressions.plt.show")
    def test_all_run(self, mock_show):
        """ test that main models run successfully.  testing for no technical code issues """

        X, Y = make_regression(n_samples=50, n_features=10, n_informative=5, n_targets=2, noise=0.5, random_state=1)
        x = RunRegression(X, Y, 'LinearRegression')
        x()

        x = RunRegression(X, Y, 'KNeighborsRegressor')
        x()

        x = RunRegression(X, Y, 'RandomForestRegressor')
        x()

        x = RunRegression(X, Y, 'DecisionTreeRegressor')
        x()
