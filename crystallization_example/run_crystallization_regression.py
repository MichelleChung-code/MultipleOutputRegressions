import pandas as pd
from Regressions import RunRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sympy import *
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# For input data schema
AGITATOR_SPEED = 'AGITATOR_SPEED'
SEED_CRYSTAL_MASS = 'SEED_CRYSTAL_MASS'
YIELD = 'YIELD'
GROWTH_RATE = 'GROWTH_RATE'
MEAN_DIAMETER = 'MEAN_DIAMETER'
CRYSTALLIZER_POWER = 'CRYSTALLIZER_POWER'

__author__ = 'Michelle Aria Chung'


class CrystallizationRegression:
    def __init__(self, input_data_path, output_path=False):
        """
        Class to run specific regression for the crystallization experiment

        Args:
            input_data_path: <str> path to the input data csv
            output_path: <str> path to write results.  If False, none are written

        """
        self.input_data = pd.read_csv(input_data_path)

        self.X = self.input_data[[AGITATOR_SPEED, SEED_CRYSTAL_MASS]]
        self.Y = self.input_data[[YIELD, GROWTH_RATE, MEAN_DIAMETER]]
        self.output_path = output_path


    def __call__(self):
        """
        Run the crystallization regressions
        """
        lb_ls = [210, 0]  # min is actually 214, 0
        ub_ls = [670, 2.2]  # so that max is 665, 2
        step_size_ls = [5, 0.2]

        # traditional way with interactions
        # doing the y values individually
        print("\n", '#' * 40, 'MULTIPLE LINEAR WITH INTERACTION TERM', '#' * 40, "\n")

        self.multiple_linear_regression_interactions()

        print("\n", '#' * 40, 'LINEAR CHAINED', '#' * 40, "\n")

        crystallization_linear_chained = RunRegression(self.X, self.Y, 'ChainedMultiOutput_Linear',
                                                       plot_individual_bool=True, output_path=self.output_path)
        crystallization_linear_chained()
        crystallization_linear_chained.plot_optimized_maximum(lb_ls, ub_ls, step_size_ls)

        print("\n", '#' * 40, 'RANDOM FOREST', '#' * 40, "\n")

        crystallization_random_forest_regression = RunRegression(self.X, self.Y, 'RandomForestRegressor',
                                                                 plot_individual_bool=True,
                                                                 output_path=self.output_path)
        crystallization_random_forest_regression()
        crystallization_random_forest_regression.plot_optimized_maximum(lb_ls, ub_ls, step_size_ls)

    def multiple_linear_regression_interactions(self):
        """
        Run the linear regressions using interaction terms
        """
        # todo move to general Regressions class after this is made more scalable.
        # Source: https://towardsdatascience.com/multiple-linear-regression-with-interactions-unveiled-by-genetic-programming-4cc325ac1b65

        X = sm.add_constant(self.X)

        for y_col in self.Y.columns:
            # generate polynomial and interaction features
            poly = PolynomialFeatures(interaction_only=True)
            X_tr = poly.fit_transform(X)
            interactions_only_df = pd.DataFrame(X_tr, columns=poly.get_feature_names()).drop(
                ['1', 'x0', 'x1', 'x2', 'x0 x1', 'x0 x2'],
                axis=1)
            X_final = pd.concat([X, interactions_only_df], axis=1)

            # refit with x1 x2 interactions
            new_model = sm.OLS(self.Y[y_col], X_final)
            new_result = new_model.fit()

            print(new_result.summary())

            # plot the new graphs
            new_Y = new_result.predict(X_final)
            CrystallizationRegression.plot_model_individual_series(model_type='LinearRegression_WithInteractions',
                                                                   X_original=self.X, Y_original=self.Y[y_col],
                                                                   y_predict=new_Y, output_path=self.output_path)

    @staticmethod
    def plot_model_individual_series(model_type, X_original, Y_original, y_predict, output_path):
        """
        Plot individual charts to depict how closely the fitted y-series follows the actual data.

        Args:
            model_type: <str> name of regression model type being run
            X_original: <pd.DataFrame> original x-series
            Y_original: <pd.Series> original y-series
            y_predict: <pd.Series> model fitted y-series
            output_path: <str> path to save figures to
        """
        # todo this is very similar to the Regression class version of plot_model, should make the Regression class one a static
        # method and then just use from there
        if isinstance(Y_original, pd.Series):
            Y_original = Y_original.to_frame()
            y_predict = y_predict.to_frame()

        num_outputs = len(y_predict.columns)
        x_ax = range(len(X_original))

        if not isinstance(Y_original, (pd.DataFrame, pd.Series)):  # todo allow for numpy array inputs as well
            raise NotImplemented

        for i in range(num_outputs):
            plt.figure(figsize=(10, 8))
            plt.plot(x_ax, Y_original.iloc[:, i], label='y_actual')
            plt.plot(x_ax, y_predict.iloc[:, i],
                     label='y_pred')  # y_predict is originally a series object
            plt.title('{type}: {y_name}'.format(type=model_type, y_name=Y_original.columns[i]))
            plt.xlabel('DATAPOINT_NUMBER')
            plt.ylabel(Y_original.columns[i])
            plt.legend()

            if output_path: # can be False
                plt.savefig(
                    os.path.join(output_path, '{}_{}_follow_fit.png'.format(model_type, Y_original.columns[i])))
            plt.show()


if __name__ == '__main__':
    # write console output to text file

    cur_path = str(Path(__file__).parents[0])
    res_path = os.path.join(cur_path, 'results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    console_out_path = os.path.join(res_path, 'crystallization_run_console_output.txt')
    sys.stdout = open(console_out_path, 'w')

    input_data_path = 'crystallization_input_data.csv'
    x = CrystallizationRegression(input_data_path, res_path)
    x()

    sys.stdout.close()
