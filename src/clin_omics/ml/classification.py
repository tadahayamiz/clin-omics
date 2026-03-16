from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int | None = 0,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_logistic_classifier(model: Pipeline, X_test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    y_score = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="y_score")
    y_pred = pd.Series(model.predict(X_test), index=X_test.index, name="y_pred")
    return y_score, y_pred
