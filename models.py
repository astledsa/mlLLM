import joblib
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier


class BaseModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class LinearRegressionModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LinearRegression()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}
    
    def predict(self, X):
        return self.model.predict(X)

class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LogisticRegression()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class RandomForestRegressorModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = RandomForestRegressor()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}

    def predict(self, X):
        return self.model.predict(X)

class RandomForestClassifierModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = RandomForestClassifier()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class GradientBoostingRegressorModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = GradientBoostingRegressor()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}

    def predict(self, X):
        return self.model.predict(X)

class GradientBoostingClassifierModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = GradientBoostingClassifier()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SVRModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = SVR()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}
    
    def predict(self, X):
        return self.model.predict(X)

    
class SVCModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = SVC(probability=True)

    def test(self):
        y_pred = self.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class KNNRegressorModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = KNeighborsRegressor()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}
    
    def predict(self, X):
        return self.model.predict(X)

class KNNClassifierModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = KNeighborsClassifier()

    def test(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MSE": mse, "R2": r2}
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)