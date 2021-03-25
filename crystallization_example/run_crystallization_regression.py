import pandas as pd
from Regressions import RunRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sympy import *
from gplearn.genetic import SymbolicRegressor

__author__ = 'Michelle Aria Chung'

# For input data schema
AGITATOR_SPEED = 'AGITATOR_SPEED'
SEED_CRYSTAL_MASS = 'SEED_CRYSTAL_MASS'
YIELD = 'YIELD'
GROWTH_RATE = 'GROWTH_RATE'
MEAN_DIAMETER = 'MEAN_DIAMETER'
CRYSTALLIZER_POWER = 'CRYSTALLIZER_POWER'

sympy_str_converter = {
    'sub': lambda x, y: x - y,
    'div': lambda x, y: x / y,
    'mul': lambda x, y: x * y,
    'add': lambda x, y: x + y,
    'neg': lambda x: -x,
    'pow': lambda x, y: x ** y,
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'inv': lambda x: 1 / x,
    'sqrt': lambda x: x ** 0.5
}


class CrystallizationRegression:
    def __init__(self, input_data_path):
        self.input_data = pd.read_csv(input_data_path)

        self.X = self.input_data[[AGITATOR_SPEED, SEED_CRYSTAL_MASS]]
        self.Y = self.input_data[[YIELD, GROWTH_RATE, MEAN_DIAMETER]]

    def __call__(self):
        lb_ls = [210, 0] # min is actually 214, 0
        ub_ls = [670, 2.2] # so that max is 665, 2
        step_size_ls = [5, 0.2]

        # traditional way with interactions
        # doing the y values individually
        self.multiple_linear_regression_interactions()

        print('=' * 50)

        crystallization_random_forest_regression = RunRegression(self.X, self.Y, 'RandomForestRegressor',
                                                                 plot_individual_bool=True)
        crystallization_random_forest_regression()
        crystallization_random_forest_regression.plot_optimized_maximum(lb_ls, ub_ls, step_size_ls)

        print('=' * 50)

        crystallization_linear_chained = RunRegression(self.X, self.Y, 'ChainedMultiOutput_Linear',
                                                       plot_individual_bool=True)
        crystallization_linear_chained()
        crystallization_linear_chained.plot_optimized_maximum(lb_ls, ub_ls, step_size_ls)

    def multiple_linear_regression_interactions(self):
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

            # genetic programming to get the analytical equation
            # function_set = ['add', 'sub', 'mul', 'div', 'cos', 'sin', 'neg', 'inv']
            # X.drop('const', axis=1, inplace=True)
            #
            # est_gp = SymbolicRegressor(population_size=5000, function_set=function_set,
            #                            generations=40, stopping_criteria=0.01,
            #                            p_crossover=0.7, p_subtree_mutation=0.1,
            #                            p_hoist_mutation=0.05, p_point_mutation=0.1,
            #                            max_samples=0.9, verbose=1,
            #                            parsimony_coefficient=0.01, random_state=0,
            #                            feature_names=X.columns)
            #
            # est_gp.fit(X, self.Y[y_col])
            # print('R2: ', est_gp.score(X, self.Y[y_col]))
            # sympy_eqn = sympify(str(est_gp._program), locals=sympy_str_converter)
            # print(sympy_eqn)


if __name__ == '__main__':
    input_data_path = 'crystallization_input_data.csv'
    x = CrystallizationRegression(input_data_path)
    x()
