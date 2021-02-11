from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np
# import json
from pprint import pprint

# https://machinelearningmastery.com/multi-output-regression-models-with-python/

dict_reg_type = {'LinearRegression': LinearRegression,
                 'KNeighborsRegressor': KNeighborsRegressor,
                 'RandomForestRegressor': RandomForestRegressor,
                 'DecisionTreeRegressor': DecisionTreeRegressor}


class RunRegression:
    def __init__(self, X, Y, model_type):
        self.X = X
        self.Y = Y
        self.model = dict_reg_type[model_type]()
        self.model_type = model_type

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

        dict_additional_output_type = {'LinearRegression': RunRegression.lin_reg_output,
                                       'KNeighborsRegressor': RunRegression.k_nearest_neighbours_output,
                                       'RandomForestRegressor': lambda *args: None,  # todo
                                       'DecisionTreeRegressor': lambda *args: None}  # todo

        return r_sq, y_predict, dict_additional_output_type[self.model_type](model)

    @staticmethod
    def lin_reg_output(model):
        return {'weights': model.coef_,
                'intercept': model.intercept_}

    @staticmethod
    def k_nearest_neighbours_output(model):
        return {'params': model.get_params()}

    def __call__(self):
        n_scores = self.evaluate_model()
        r_sq, y_predict, additional_output = self.run_regression_model()

        print('Mean Absolute Error, mean (std): {:.3f} ({:.3f})'.format(np.mean(n_scores), np.std(n_scores)))
        print('r^2: {:.3f}'.format(r_sq))
        print('Predicted y-series: \n{}'.format(y_predict))

        # the below line only works for primative types
        # print(json.dumps(additional_output, indent=4, sort_keys=True))

        pprint(additional_output)
