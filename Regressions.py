from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

dict_reg_type = {'LinearRegression': LinearRegression,
                 'KNeighborsRegressor': KNeighborsRegressor,
                 'RandomForestRegressor': RandomForestRegressor,
                 'DecisionTreeRegressor': DecisionTreeRegressor}


class RunRegression:
    def __init__(self, X, Y, model_type):
        self.X = X
        self.Y = Y
        self.model = dict_reg_type[model_type]()

    def run_regression_model(self):
        # fit the model
        model = self.model.fit(self.X, self.Y)
        r_sq = model.score(self.X, self.Y)
        y_predict = model.predict(self.X)

        return r_sq, y_predict

    def __call__(self):
        r_sq, y_predict = self.run_regression_model()

        print(r_sq)
        print(y_predict)
