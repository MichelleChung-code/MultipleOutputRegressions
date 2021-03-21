import os
from pathlib import Path
import pandas as pd
from Regressions import RunRegression

__author__ = 'Michelle Aria Chung'

# For input data schema
AGITATOR_SPEED = 'AGITATOR_SPEED'
SEED_CRYSTAL_MASS = 'SEED_CRYSTAL_MASS'
YIELD = 'YIELD'
GROWTH_RATE = 'GROWTH_RATE'
MEAN_DIAMETER = 'MEAN_DIAMETER'
CRYSTALLIZER_POWER = 'CRYSTALLIZER_POWER'


class CrystallizationRegression:
    def __init__(self, input_data_path):
        self.input_data = pd.read_csv(input_data_path)

        self.X = self.input_data[[AGITATOR_SPEED, SEED_CRYSTAL_MASS]]
        self.Y = self.input_data[[YIELD, GROWTH_RATE, MEAN_DIAMETER, CRYSTALLIZER_POWER]]

    def plot_original_data(self):
        # todo generate different x vs y plots, while keeping other x vars constant
        pass

    def __call__(self):
        crystallization_linear_regression = RunRegression(self.X, self.Y, 'LinearRegression', plot_individual_bool=True)
        crystallization_linear_regression()

        crystallization_random_forest_regression = RunRegression(self.X, self.Y, 'RandomForestRegressor',
                                                                 plot_individual_bool=True)
        crystallization_random_forest_regression()

        # crystallization_linear_direct = RunRegression(self.X, self.Y, 'ChainedMultiOutput_Linear', plot_individual_bool=True)
        # crystallization_linear_direct()

if __name__ == '__main__':
    input_data_path = 'crystallization_input_data.csv'
    x = CrystallizationRegression(input_data_path)
    x()
