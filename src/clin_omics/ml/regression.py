from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_linear_regressor(model: Pipeline, X_test: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X_test), index=X_test.index, name="y_pred")
