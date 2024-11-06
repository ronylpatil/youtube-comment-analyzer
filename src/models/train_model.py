# implement model training here Exp-2
# Exp-1 Baseline model
# Exp-2 Bow vs TF-IDF
# Exp-3 Best from Exp-2 vs Unigram vs Bigram vs Trigram
# Exp-4 ML Algo's
# Exp-5 Tune Best Model from Exp-4

# create function jisme hum exp ka naam pass karenge or vo humko sare parameters us experiment k under track karega

import yaml
import mlflow
import dagshub  # type: ignore
import pathlib
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.logger import infologger
from src.features.build_features import loadData
from src.visualization.visualize import conf_matrix


def run_experiment(
    df: pd.DataFrame,
    max_features: int,
    model_params: dict,
    path: str,  # path where confusion matrix will be saved
    test_size: float,
    experiment_name: str,
    vectorizer_type: str,
    n_gram: Tuple,
    model_name: str,
    n_gram_name: str,
    model_dir: str,
) -> None:
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"], test_size=test_size
    )

    X_train_vect, X_test_vect = feature_extraction(
        X_train=X_train,
        X_test=X_test,
        vectorizer_type=vectorizer_type,
        ngram_range=n_gram,
        max_features=max_features,
    )

    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # set tags for the experiment
        mlflow.set_tag(
            "mlflow.runName", f"{vectorizer_type}_{n_gram_name}_{model_name}"
        )
        mlflow.set_tag("experiment_type", f"{experiment_name}")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # add a description
        mlflow.set_tag(
            "description",
            f"RandomForest with {vectorizer_type}, ngram_range={n_gram_name}, max_features={max_features}",
        )

        # log vectorizer parameters
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", tuple(n_gram))
        mlflow.log_param("vectorizer_max_features", max_features)

        # log model hyper-parameters
        for i, j in model_params.items():
            mlflow.log_param(i, j)

        model_signature = infer_signature(X_train_vect, y_train)

        # train the model
        y_pred, model = train_model(
            X_train_vect,
            y_train,
            X_test_vect,
            params=model_params,
            model_name="Random_Forest",
        )

        # log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # log confusion matrix
        cm_path = conf_matrix(y_test, y_pred, path=path)
        print(f"cm_path: {cm_path}")
        # mlflow.log_artifact(cm_path, "confusion_matrix")

        # save model to local
        joblib.dump(model, model_dir)

        # log the model
        mlflow.sklearn.log_model(
            model, f"random_forest_model_{vectorizer_type}_{n_gram_name}", signature = model_signature
        )


def feature_extraction(
    vectorizer_type: str, max_features: int, ngram_range: Tuple, X_train: pd.DataFrame, X_test: pd.DataFrame
):
    if vectorizer_type == "bow":
        try:
            cv = CountVectorizer(
                max_features=max_features, ngram_range=tuple(ngram_range)
            )
            X_train_vect = cv.fit_transform(X_train)
            X_test_vect = cv.transform(X_test)
            infologger.info(f"features extracted (bow) successfully...")
        except Exception as e:
            infologger.info(f"some issue in bow. Error: {e}")
        else:
            return X_train_vect, X_test_vect
    elif vectorizer_type == "tfidf":
        try:
            tfidf = TfidfVectorizer(
                max_features=max_features, ngram_range=tuple(ngram_range)
            )
            X_train_vect = tfidf.fit_transform(X_train)
            X_test_vect = tfidf.transform(X_test)
            infologger.info(f"features extracted (tfidf) successfully...")
        except Exception as e:
            infologger.info(f"some issue in tfidf. Error: {e}")
        else:
            return X_train_vect, X_test_vect


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict,
    model_name: str,
) -> pd.Series:
    # train multiple models
    if model_name == "Random_Forest":
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred, model


def main() -> None:
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    dagshub.init(
        repo_owner=params["train_model"]["repo_owner"],
        repo_name=params["train_model"]["repo_name"],
        mlflow=True,
    )

    df = loadData(f"{home_dir}/data/processed/clean_data.csv").dropna()

    run_experiment(
        df=df,
        max_features=params["build_features"]["max_features"],
        model_name=params["train_model"]["model_name"],
        experiment_name=params["train_model"]["experiment_name"],
        vectorizer_type=params["build_features"]["vectorizer_type"],
        n_gram=params["build_features"]["n_gram"],
        model_params=params["train_model"]["hyperparams"]["Random_Forest"],
        path=f"{home_dir}/figures",
        test_size=params["train_model"]["test_size"],
        n_gram_name=params["build_features"]["n_gram_name"],
        model_dir=f"{home_dir}/models/baseline_model.joblib"
    )

    features = feature_extraction(
        "bow", max_features=10000, ngram_range=(1, 2), data=df["clean_text"]
    )
    # print(params["train_model"]["experiment_name"])


if __name__ == "__main__":
    infologger.info("train_model.py as __main__")
    main()

# confusion matrix function not returning path check it
# implement script to fetch the registered model from dagshub