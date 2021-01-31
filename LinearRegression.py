from sklearn.linear_model import LinearRegression


class RunLinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def linear_regression_model(self):
        # fit the model
        model = LinearRegression().fit(self.X, self.Y)
        r_sq = model.score(self.X, self.Y)
        y_predict = model.predict(self.X)

        return r_sq, y_predict

    def __call__(self):
        r_sq, y_predict = self.linear_regression_model()

        print(r_sq)
        print(y_predict)
