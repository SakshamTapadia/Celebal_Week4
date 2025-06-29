"""
models/model_selector.py: Automates model selection and hyperparameter tuning.
"""

from typing import Tuple, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import numpy as np

from models.traditional_models import LogisticRegressionModel, RandomForestModel, SVMModel, XGBoostModel
from models.ensemble_models import VotingEnsembleModel, StackingEnsembleModel


class ModelSelector:
    def __init__(self, model_name: str, param_grid: dict, cv: int = 5, search_type: str = "grid"):
        self.model_name = model_name
        self.param_grid = param_grid
        self.cv = cv
        self.search_type = search_type

    def get_model_class(self):
        mapping = {
            "LogisticRegression": LogisticRegressionModel,
            "RandomForest": RandomForestModel,
            "SVM": SVMModel,
            "XGBoost": XGBoostModel,
            "VotingEnsemble": VotingEnsembleModel,
            "StackingEnsemble": StackingEnsembleModel
        }
        return mapping[self.model_name]

    def tune_model(self, X: Any, y: Any):
        model_class = self.get_model_class()

        if self.search_type == "grid":
            base_model = model_class()
            search = GridSearchCV(
                estimator=base_model.model,
                param_grid=self.param_grid,
                scoring='f1',
                cv=self.cv,
                n_jobs=-1
            )
            search.fit(X, y)
            best_params = search.best_params_
            best_model = model_class(**best_params)
        elif self.search_type == "random":
            base_model = model_class()
            search = RandomizedSearchCV(
                estimator=base_model.model,
                param_distributions=self.param_grid,
                scoring='f1',
                cv=self.cv,
                n_iter=20,
                n_jobs=-1
            )
            search.fit(X, y)
            best_params = search.best_params_
            best_model = model_class(**best_params)
        else:
            raise ValueError("Unsupported search_type. Use 'grid' or 'random'.")

        best_model.train(X, y)
        return best_model

    def tune_with_optuna(self, X: Any, y: Any):
        def objective(trial):
            if self.model_name == "XGBoost":
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    use_label_encoder=False,
                    eval_metric="logloss"
                )
            else:
                raise NotImplementedError("Only XGBoost is supported with Optuna for now.")

            from sklearn.model_selection import cross_val_score
            score = cross_val_score(model, X, y, scoring="f1", cv=self.cv, n_jobs=-1)
            return score.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)
        best_params = study.best_params
        model_class = self.get_model_class()
        best_model = model_class(**best_params)
        best_model.train(X, y)
        return best_model
