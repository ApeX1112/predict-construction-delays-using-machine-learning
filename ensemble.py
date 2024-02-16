import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class ensemble():
    def __init__(self,ensemble1,ensemble2,ensemble3,ensemble4,X_train,y_train,X_test,y_test):
        self.ensemble1=ensemble1
        self.ensemble2=ensemble2
        self.ensemble3=ensemble3
        self.ensemble4=ensemble4
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.meta_model=LinearRegression()

    def _fit_models(self) -> None:
        self.ensemble1.fit(self.X_train,self.y_train)
        self.ensemble2.fit(self.X_train,self.y_train)
        self.ensemble3.fit(self.X_train,self.y_train)
        self.ensemble4.fit(self.X_train,self.y_train)
    
    def _meta_features(self):
        self._fit_models()
        mf=np.column_stack([
            self.ensemble1.predict(self.X_test),
            self.ensemble2.predict(self.X_test),
            self.ensemble3.predict(self.X_test),
            self.ensemble4.predict(self.X_test)
        ])

        return mf
    
    def meta_features(self,new_X_test):
        self._fit_models()
        mf=np.column_stack([
            self.ensemble1.predict(new_X_test),
            self.ensemble2.predict(new_X_test),
            self.ensemble3.predict(new_X_test),
            self.ensemble4.predict(new_X_test)
        ])

        return mf

    def train_meta_model(self):
        self._fit_models()
        X_meta=self._meta_features()
        y_meta=self.y_test
        self.meta_model.fit(X_meta,y_meta)


    def r2_score_models(self):
        self._fit_models()
        return [
            r2_score(self.ensemble1.predict(self.X_test),self.y_test),
            r2_score(self.ensemble2.predict(self.X_test),self.y_test),
            r2_score(self.ensemble3.predict(self.X_test),self.y_test),
            r2_score(self.ensemble4.predict(self.X_test),self.y_test)
        ]


    def make_prediction(self,X_new):
        self.train_meta_model()
        X_new_meta=self.meta_features(X_new)
        return self.meta_model.predict(X_new_meta)

    
#////////////////////////TEST////////////////////////////////////////////////////////////////
    
'''

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


ensemble1 = DecisionTreeRegressor(random_state=42)
ensemble2 = RandomForestRegressor(n_estimators=10, random_state=42)  # Use fewer estimators for faster testing
ensemble3 = DecisionTreeRegressor(max_depth=5, random_state=42)
ensemble4 = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)


ensemble_model = ensemble(ensemble1, ensemble2, ensemble3, ensemble4, X_train, y_train, X_test, y_test)


ensemble_model.train_meta_model()


r2_scores = ensemble_model.r2_score_models()
print("R^2 scores of the base models:", r2_scores)


predictions = ensemble_model.make_prediction(X_test)  

print("Predictions for{}new data:".format(predictions.shape),predictions)


'''