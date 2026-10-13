"""The three classifiers compared in this project."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build(seed: int = 42) -> dict:
    return {
        # Baseline: linear model. Needs imputed, scaled inputs.
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        # Small neural network: two hidden layers, early stopping.
        "Neural Network (MLP)": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                          max_iter=200, random_state=seed),
        ),
        # Gradient-boosted trees: handles missing values natively.
        "XGBoost": XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", eval_metric="auc",
            random_state=seed, n_jobs=-1,
        ),
    }
