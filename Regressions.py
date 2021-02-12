from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np
# import json
from pprint import pprint
import matplotlib.pyplot as plt

# https://machinelearningmastery.com/multi-output-regression-models-with-python/

dict_reg_type = {'LinearRegression': LinearRegression,
                 'KNeighborsRegressor': KNeighborsRegressor,
                 'RandomForestRegressor': RandomForestRegressor,
                 'DecisionTreeRegressor': DecisionTreeRegressor}


class RunRegression:
    def __init__(self, X, Y, model_type):
        """
        Initiate the RunRegression class

        Args:
            X: <np.Array> input independent variable(s) data
            Y: <np.Array> input dependent variable(s) data
            model_type: <str> regression type, must be one in dict_reg_type variable above
        """
        self.X = X
        self.Y = Y
        self.model = dict_reg_type[model_type]()
        self.model_type = model_type

    def evaluate_model(self):
        """
        Evaluate the regression method using 10-fold cross validation

        Returns:
            n_scores: <np.Array> absolute values of the evaluated scores for each cross validation run
        """
        # Evaluate the multi-output regression using k-fold cross-validation

        # 10-fold cross-val
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(self.model, self.X, self.Y, scoring='neg_mean_absolute_error', cv=cv)

        return np.abs(n_scores)

    def run_regression_model(self):
        """
        Run the specified regression type and fit the data

        Returns:
            r_sq: <float> r^2 of the prediction
            y_predict: <np.Array> predicted y-series of the regression model
            additional_output: <dict> containing regression-type specific additional output as specified by functions in
            dict_additional_output_type
        """
        # fit the model
        model = self.model.fit(self.X, self.Y)
        r_sq = model.score(self.X, self.Y)
        y_predict = model.predict(self.X)

        dict_additional_output_type = {'LinearRegression': RunRegression.lin_reg_output,
                                       'KNeighborsRegressor': RunRegression.get_params_output,
                                       'RandomForestRegressor': RunRegression.get_params_output,
                                       'DecisionTreeRegressor': RunRegression.get_params_output}

        return r_sq, y_predict, dict_additional_output_type[self.model_type](model)

    @staticmethod
    def lin_reg_output(model):
        """
        Get the weights and y-intercept of the linear regression model

        Args:
            model: <sklearn.linear_model._base.LinearRegression> fitted linear regression model

        Returns:
            <dict> containing the resulting weights and intercept
        """
        return {'weights': model.coef_,
                'intercept': model.intercept_}

    @staticmethod
    def get_params_output(model):
        """
        Get general .get_params() results for additional model information

        Args:
            model: <sklearn.linear_model._base.{model_type}> to run general .get_params() on

        Returns:
            <dict> containing the results of the .get_params() call
        """
        return {'params': model.get_params()}

    def plot_model(self, y_predict):
        """
        Product plot comparing the actual and predicted values to better see how closely the data was fitted

        Args:
            y_predict: <np.Array> predicted y-series of the regression model
        """
        row, num_outputs = y_predict.shape
        x_ax = range(len(self.X))

        for i in range(num_outputs):
            plt.plot(x_ax, self.Y[:, i], label='y{}_actual'.format(i + 1))
            plt.plot(x_ax, y_predict[:, i], label='y{}_pred'.format(i + 1))

        plt.title(self.model_type)
        plt.legend()
        plt.show()

    def __call__(self):
        """ Run the regression class """
        n_scores = self.evaluate_model()
        r_sq, y_predict, additional_output = self.run_regression_model()
        print(self.model_type)
        print('Mean Absolute Error, mean (std): {:.3f} ({:.3f})'.format(np.mean(n_scores), np.std(n_scores)))
        print('r^2: {:.3f}'.format(r_sq))
        print('Predicted y-series: \n{}'.format(y_predict[:10]))

        # the below line only works for primative types
        # print(json.dumps(additional_output, indent=4, sort_keys=True))

        pprint(additional_output)

        self.plot_model(y_predict)
