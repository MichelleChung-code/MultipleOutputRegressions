from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np
# import json
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import warnings
from sklearn.exceptions import ConvergenceWarning
import itertools
import os
from pathlib import Path

# https://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms
DirectMultiOutput = 'DirectMultiOutput'
ChainedMultiOutput = 'ChainedMultiOutput'

dict_reg_type = {'LinearRegression': LinearRegression,
                 'KNeighborsRegressor': KNeighborsRegressor,
                 'RandomForestRegressor': RandomForestRegressor,
                 'DecisionTreeRegressor': DecisionTreeRegressor,
                 DirectMultiOutput + '_Linear': LinearSVR,
                 ChainedMultiOutput + '_Linear': LinearRegression}

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class RunRegression:
    def __init__(self, X, Y, model_type, plot_individual_bool=False, plot_summary_one_bool=False):
        """
        Initiate the RunRegression class

        Args:
            X: <np.Array> input independent variable(s) data
            Y: <np.Array> input dependent variable(s) data
            model_type: <str> regression type, must be one in dict_reg_type variable above
            plot_individual_bool: <bool> whether to also plot the individual x vs y series
            plot_summary_one_bool: <bool> whether to plot the summary comparison chart on one graph or separate the
            different y-series into their own charts
        """
        self.X = X
        self.Y = Y
        self.model = dict_reg_type[model_type]()
        self.model_type = model_type
        self.plot_individual_bool = plot_individual_bool
        self.plot_summary_one_bool = plot_summary_one_bool

        # check whether the type involves a multi-output wrapper
        multi_output_wrapper = model_type.split('_')[0]
        if multi_output_wrapper in [DirectMultiOutput, ChainedMultiOutput]:
            self.model = MultiOutputRegressor(
                self.model) if multi_output_wrapper == DirectMultiOutput else RegressorChain(self.model)

    def evaluate_model(self):
        """
        Evaluate the regression method using 10-fold cross validation

        Returns:
            n_scores: <np.Array> absolute values of the evaluated scores for each cross validation run
        """
        # Evaluate the multi-output regression using k-fold cross-validation

        # 10-fold cross-val

        num_splits = 10 if len(self.X) >= 10 else len(self.X)  # defaults to 10 as long as there are at least 10 samples
        cv = RepeatedKFold(n_splits=num_splits, n_repeats=3, random_state=1)
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

        if isinstance(self.Y, pd.DataFrame):
            r_sq_indv_ls = [r2_score(self.Y.iloc[:, i], y_predict[:, i]) for i in range(len(self.Y.columns))]
        else:
            r_sq_indv_ls = [r2_score(self.Y[:, i], y_predict[:, i]) for i in range(len(self.Y.columns))]

        additional_output = dict_additional_output_type.get(self.model_type, False)

        if additional_output:
            additional_output = additional_output(model)

        return r_sq, r_sq_indv_ls, y_predict, additional_output

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

    def plot_optimized_maximum(self, lb_ls, ub_ls, step_size_ls):
        """
        Run different combinations of inputs and predict the outputs using the current model.  Plots the results and
        prints the optimal input conditions (resulting in the max output values)

        Args:
            lb_ls: <list> of lower bound values for each input
            ub_ls: <list> of upper bound values for each input
            step_size_ls: <list> of step size to run the combinations for, between lower and upper bound values, for
            each input
        """
        assert len(lb_ls) == len(ub_ls)  # lower and upper bound lists must have the same number of elements
        if not isinstance(self.Y, pd.DataFrame):
            raise NotImplemented  # todo implement this piece

        values_ls = []
        # get combinations of possible settings
        for i in range(len(lb_ls)):
            values_ls.append([*np.arange(lb_ls[i], ub_ls[i], step_size_ls[i])])

        combinations_ls = list(itertools.product(*values_ls))
        new_X_df = pd.DataFrame(combinations_ls, columns=self.X.columns)
        predicted_y = self.model.predict(new_X_df)

        row, num_outputs = self.model.predict(new_X_df).shape
        cur_path = str(Path(__file__).parents[0])
        max_values = np.argmax(predicted_y, axis=0)
        optimal_vals_dict = {}
        for i in range(len(max_values)):
            optimal_vals_dict[self.Y.columns[i]] = {
                'conditions {}'.format(list(new_X_df.columns)): new_X_df.iloc[max_values[i]].to_list(),
                'value': predicted_y[max_values[i], i]}

        print('OPTIMAL CONDITIONS: {}'.format(self.model_type))
        full_runs_df = pd.DataFrame(predicted_y, columns=self.Y.columns)
        full_runs_df = pd.concat([new_X_df, full_runs_df], axis=1)

        full_runs_df.to_csv(
            os.path.join(cur_path, 'crystallization_example/results/{}_full_runs.csv'.format(self.model_type)))

        pprint(optimal_vals_dict)

        for i in range(num_outputs):
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(new_X_df.iloc[:, 0], new_X_df.iloc[:, 1], predicted_y[:, 1], c=predicted_y[:, 1],
                             cmap=plt.get_cmap('BuPu'))
            ax.set_xlabel(self.X.columns[0])
            ax.set_ylabel(self.X.columns[1])
            ax.set_zlabel(self.Y.columns[i])
            ax.set_title('{}: {}'.format(self.model_type, self.Y.columns[i]))
            # fig.set_size_inches(12, 8)
            fig.colorbar(p, ax=ax)
            plt.savefig(os.path.join(cur_path, 'crystallization_example/results/{}_{}.png'.format(self.model_type,
                                                                                                  self.Y.columns[i])))

    def plot_model(self, y_predict, one_plot=True):
        """
        Product plot comparing the actual and predicted values to better see how closely the data was fitted
        Shows how closely the y data follows, per data point

        Args:
            y_predict: <np.Array> predicted y-series of the regression model
            one_plot: <bool> if True plot all resulting y-series on one graph.  If False, plot all on individual graphs.
        """
        row, num_outputs = y_predict.shape
        x_ax = range(len(self.X))
        cur_path = str(Path(__file__).parents[0])

        for i in range(num_outputs):
            plt.figure(figsize=(10, 8))
            if not one_plot:
                if not isinstance(self.Y, pd.DataFrame):
                    raise NotImplemented  # todo implement this case

            if isinstance(self.Y, pd.DataFrame):
                plt.plot(x_ax, np.array(self.Y.iloc[:, i]), label='y_actual')
            else:
                plt.plot(x_ax, self.Y[:, i], label='y_actual')
            plt.plot(x_ax, y_predict[:, i], label='y_pred')

            if not one_plot:
                plt.title('{}: {}'.format(self.model_type, self.Y.columns[i]))
                plt.legend()
                plt.xlabel('DATAPOINT_NUMBER')
                plt.ylabel(self.Y.columns[i])
                plt.savefig(os.path.join(cur_path, 'crystallization_example/results/{}_{}_follow_fit.png'.format(self.model_type,
                                                                                                      self.Y.columns[
                                                                                                          i])))
                plt.show()

        if one_plot:
            plt.title(self.model_type)
            plt.legend()
            plt.show()

    def plot_model_individual_series(self, y_predict):
        """
        Plot original x-values against original y-values and newly predicted y-values.

        Args:
            y_predict: <pd.Data
        """
        row, num_outputs = y_predict.shape
        num_inputs = len(self.X.columns)

        if not isinstance(self.Y, pd.DataFrame):  # todo allow for numpy array inputs as well
            raise NotImplemented

        for i in range(num_outputs):
            for j in range(num_inputs):
                plt.plot(self.X.iloc[:, j], self.Y.iloc[:, i], label='y_actual')
                plt.plot(self.X.iloc[:, j], y_predict[:, i], label='y_pred')
                plt.title('{type}: {y_name} vs. {x_name}'.format(type=self.model_type, y_name=self.Y.columns[i],
                                                                 x_name=self.X.columns[j]))
                plt.xlabel(self.X.columns[j])
                plt.ylabel(self.Y.columns[i])
                plt.legend()
                plt.show()

    def __call__(self):
        """ Run the regression class """
        n_scores = self.evaluate_model()
        r_sq, r_sq_indv_ls, y_predict, additional_output = self.run_regression_model()
        print(self.model_type)
        print('Mean Absolute Error, mean (std): {:.3f} ({:.3f})'.format(np.mean(n_scores), np.std(n_scores)))
        print('r^2: {:.3f}'.format(r_sq))
        print('r^2 individual: {}'.format(r_sq_indv_ls))
        print('Predicted y-series: \n{}'.format(y_predict[:10]))

        if additional_output:
            print('Additional Output:')
            pprint(additional_output)

        if self.plot_individual_bool:
            self.plot_model_individual_series(y_predict)

        self.plot_model(y_predict, self.plot_summary_one_bool)
