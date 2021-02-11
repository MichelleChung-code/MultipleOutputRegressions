from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np

dict_reg_type = {'LinearRegression': LinearRegression,
                 'KNeighborsRegressor': KNeighborsRegressor,
                 'RandomForestRegressor': RandomForestRegressor,
                 'DecisionTreeRegressor': DecisionTreeRegressor}


class RunRegression:
    def __init__(self, X, Y, model_type):
        self.X = X
        self.Y = Y
        self.model = dict_reg_type[model_type]()

    def evaluate_model(self):
        # Evaluate the multioutput regression using k-fold cross-validation

        # 10-fold cross-val
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(self.model, self.X, self.Y, scoring='neg_mean_absolute_error', cv=cv)

        return np.abs(n_scores)

    def run_regression_model(self):
        # fit the model
        model = self.model.fit(self.X, self.Y)
        r_sq = model.score(self.X, self.Y)
        y_predict = model.predict(self.X)

        return r_sq, y_predict

    def __call__(self):
        n_scores = self.evaluate_model()
        r_sq, y_predict = self.run_regression_model()

        print('Mean Absolute Error, mean (std): {:.3f} ({:.3f})'.format(np.mean(n_scores), np.std(n_scores)))
        print('r^2: {:.3f}'.format(r_sq))
        print('Predicted y-series: \n{}'.format(y_predict))
