import os

import numpy as np
import orjson
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import MLTSVM, MLkNN
from typeguard import typechecked

ROOT = "data/"
MODEL_DIR = "models/"
MODELS = ["mlknn", "mltsvm"]
FEATURES = ["aac", "cht", "pcp", "combination"]


@typechecked
class Pipeline:
    def __init__(self) -> None:
        print("Initializing pipeline...")
        self.data: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = dict()
        self.hyperparams_search = {
            "mlknn": {
                "k": range(3, 15),
                "s": [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4],
            },
            "mltsvm": {
                "c_k": [2**i for i in range(-5, 6)],
                "sor_omega": [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
                "lambda_param": [2**i for i in range(-5, 2)],
            },
        }
        self.hyperparams: dict = {}
        self._create_directories()
        self._load_data()

    def _create_directories(self) -> None:
        print("Creating directories...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        for model in MODELS:
            os.makedirs(os.path.join(MODEL_DIR, model), exist_ok=True)

    def _load_data(self) -> None:
        for feature in FEATURES:
            print(f"Loading data for {feature}...")
            self.data[feature] = {"train": [], "test": []}

            train_data = sorted(
                os.listdir(os.path.join(ROOT, "train", feature)),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            test_data = sorted(
                os.listdir(os.path.join(ROOT, "test", feature)),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )

            for file in train_data:
                self.data[feature]["train"].append(
                    self._transform_data(
                        pd.read_csv(os.path.join(ROOT, "train", feature, file))
                    )
                )
            for file in test_data:
                self.data[feature]["test"].append(
                    self._transform_data(
                        pd.read_csv(os.path.join(ROOT, "test", feature, file))
                    )
                )

    def _transform_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df["anti_gram_positive"] = df["Activity"].str.contains("positive").astype(int)
        df["anti_gram_negative"] = df["Activity"].str.contains("negative").astype(int)
        df.drop(columns=["Activity", "Sequence"], inplace=True)

        return (
            df.drop(columns=["anti_gram_positive", "anti_gram_negative"]).values,
            df[["anti_gram_positive", "anti_gram_negative"]].values,
        )

    def _estimate_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        scoring: list = ["accuracy", "f1_macro"],
    ) -> tuple[dict[str, float], float]:
        model = MLkNN() if model_name == "mlknn" else MLTSVM()

        clf = GridSearchCV(
            model,
            self.hyperparams_search[model_name],
            scoring=scoring,
            refit=scoring[0],
            cv=MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=1,
        )
        clf.fit(X, y)
        return clf.best_params_, clf.best_score_

    def _run_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        hyperparams: dict,
    ) -> np.ndarray:
        print(f"Running {model_name} with hyperparameters: {hyperparams}")
        model: MLkNN | MLTSVM
        predictions: np.ndarray
        if model_name == "mlknn":
            model = MLkNN(**hyperparams)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test).toarray()
        else:
            model = MLTSVM(**hyperparams)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        return predictions

    def _evaluate(
        self, predictions: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        metrics = dict()

        metrics["hamming_loss"] = hamming_loss(y_test, predictions)
        metrics["exact_match_ratio"] = accuracy_score(y_test, predictions)
        metrics["accuracy"] = 1 - hamming_loss(y_test, predictions)
        metrics["f1_score_micro"] = f1_score(y_test, predictions, average="micro")
        metrics["f1_score_macro"] = f1_score(y_test, predictions, average="macro")
        metrics["precision"] = precision_score(y_test, predictions, average="micro")
        metrics["recall"] = recall_score(y_test, predictions, average="micro")
        metrics["jaccard_score"] = jaccard_score(y_test, predictions, average="micro")

        return metrics

    def _load_hyperparameters(self, model: str) -> bool:
        print(f"Loading hyperparameters for {model}...")
        try:
            with open(os.path.join(MODEL_DIR, model, "hyperparameters.json"), "r") as f:
                self.hyperparams[model] = orjson.loads(f.read())
        except FileNotFoundError:
            return False
        return True

    def _save_hyperparameters(self, model: str) -> None:
        if self.hyperparams.get(model) is None:
            return

        print(f"Saving hyperparameters for {model}...")
        with open(os.path.join(MODEL_DIR, model, "hyperparameters.json"), "wb") as f:
            f.write(orjson.dumps(self.hyperparams[model], option=orjson.OPT_INDENT_2))

        with pd.ExcelWriter(
            os.path.join(MODEL_DIR, model, "hyperparameters.xlsx")
        ) as writer:
            for feature, parameters in self.hyperparams[model].items():
                df = pd.DataFrame(parameters)
                df.to_excel(writer, sheet_name=feature, index=False)

    def _save_predictions(
        self, predictions: dict[str, list[dict[str, list]]], model
    ) -> None:
        print(f"Saving predictions for {model}...")
        with pd.ExcelWriter(
            os.path.join(MODEL_DIR, model, "predictions.xlsx")
        ) as writer:
            for feature, preds_list in predictions.items():
                for i, preds in enumerate(preds_list):
                    df = pd.DataFrame(preds)

                    df["exact_match_ratio"] = (
                        (
                            df["actual_anti_gram_positive"]
                            == df["predicted_anti_gram_positive"]
                        )
                        & (
                            df["actual_anti_gram_negative"]
                            == df["predicted_anti_gram_negative"]
                        )
                    ).astype(int)
                    df["accuracy_anti_gram_positive"] = (
                        df["actual_anti_gram_positive"]
                        == df["predicted_anti_gram_positive"]
                    ).astype(int)
                    df["accuracy_anti_gram_negative"] = (
                        df["actual_anti_gram_negative"]
                        == df["predicted_anti_gram_negative"]
                    ).astype(int)

                    df.to_excel(writer, sheet_name=f"{feature}_{i+1}", index=False)

    def _save_results(
        self, results: dict[str, list[dict[str, float]]], model: str
    ) -> None:
        print(f"Saving results for {model}...")
        with pd.ExcelWriter(os.path.join(MODEL_DIR, model, "results.xlsx")) as writer:
            for feature, metrics in results.items():
                df = pd.DataFrame(metrics)
                df.to_excel(writer, sheet_name=feature, index=False)

    def run(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError(f"Unsupported model: {model}")
        if not self._load_hyperparameters(model):
            hyperparams = {feature: [] for feature in FEATURES}
            for feature in FEATURES:
                for i, (X_train, y_train) in enumerate(self.data[feature]["train"]):
                    print(f"Estimating hyperparameters for {feature} dataset {i}...")
                    params, _ = self._estimate_hyperparameters(X_train, y_train, model)
                    print(params)
                    hyperparams[feature].append(params)
            self.hyperparams[model] = hyperparams
            self._save_hyperparameters(model)
        self._save_hyperparameters(model)
        if self.hyperparams is None:
            return

        results = {FEATURE: [] for FEATURE in FEATURES}
        final_predictions = {FEATURE: [] for FEATURE in FEATURES}

        for feature in FEATURES:
            for i in range(len(self.data[feature]["train"])):
                print(f"Running {model} on {feature} dataset {i}...")
                X_train, y_train = self.data[feature]["train"][i]
                X_test, y_test = self.data[feature]["test"][i]
                predictions: np.ndarray
                predictions = self._run_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    self.hyperparams[model][feature][i],
                )

                metrics = self._evaluate(predictions, y_test)
                results[feature].append(metrics)

                sequences = pd.read_csv(
                    os.path.join(ROOT, "test", feature, f"test_{feature}_{i+1}.csv")
                )["Sequence"].values

                final_predictions[feature].append(
                    {
                        "Sequence": sequences.tolist(),
                        "actual_anti_gram_positive": y_test[:, 0].tolist(),
                        "actual_anti_gram_negative": y_test[:, 1].tolist(),
                        "predicted_anti_gram_positive": predictions[:, 0].tolist(),
                        "predicted_anti_gram_negative": predictions[:, 1].tolist(),
                    }
                )

        self._save_results(results, model)
        self._save_predictions(final_predictions, model)
