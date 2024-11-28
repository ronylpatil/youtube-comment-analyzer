# Exp-1 Baseline Model (training baseline models for benchmark)
# Exp-2 BoW vs TF-IDF vs Word2vec (bow vs tfidf vs word2vec vectorization comparision with Unigram/Bigram/Trigram)
# Exp-3 Max Features (select max features 1k, 2k, 3k, 5k, 7k, 10k)
# Exp-4 Handling Imbalanced Data (try out SMOTE, ADASYN)
# Exp-5 Model Tunning (fine tune best model from Exp-4 using optuna)

import yaml
import mlflow
import joblib
import dagshub  # type: ignore
import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb  # type: ignore
import lightgbm as lgb  # type: ignore
from typing import Tuple
from datetime import datetime
from src.logger import infologger
from gensim.models import Word2Vec
from catboost import CatBoostClassifier  # type: ignore
from src.features.build_features import loadData
from mlflow.models.signature import infer_signature
from src.visualization.visualize import conf_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

infologger.info("*** Executing: train_model.py ***")


def run_experiment(
    df: pd.DataFrame,
    max_features: int,
    model_params: dict,
    path: str,  # path where confusion matrix will be saved
    test_size: float,
    experiment_name: str,
    experiment_description: str,
    vectorizer_type: str,
    n_gram: Tuple,
    model_name: str,
    model_dir: str,
    vectorizer_path: str,
) -> None:

    infologger.info("run_experiment started...")

    if n_gram == [1, 1]:
        n_gram_name = "unigram"
    elif n_gram == [1, 2]:
        n_gram_name = "bigram"
    elif n_gram == [1, 3]:
        n_gram_name = "trigram"

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"], test_size=test_size
    )
    infologger.info(f"data splited for training and testing. test_size: {test_size}")

    # convert [-1,0,1] to [0,1,2]
    target_map = {-1: 0, 0: 1, 1: 2}
    y_train = y_train.map(target_map).astype(float)
    y_test = y_test.map(target_map).astype(float)
    infologger.info(f"target varible mapped successfully. mapping: {target_map}")

    # convert text to vectors
    X_train_vect, X_test_vect = feature_extraction(
        X_train=X_train,
        X_test=X_test,
        vectorizer_type=vectorizer_type,
        ngram_range=n_gram,
        max_features=max_features,
        vectorizer_path=vectorizer_path,
    )
    infologger.info("input data vectorized successfully")

    # set mlflow experiment name
    mlflow.set_experiment(experiment_name)
    # experiment_description = f"unigram vs bigram vs trigram"
    experiment_description = experiment_description
    mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

    with mlflow.start_run() as run:

        if max_features == 1000:
            max_f = "1k"
        elif max_features == 2000:
            max_f = "2k"
        elif max_features == 3000:
            max_f = "3k"
        elif max_features == 5000:
            max_f = "5k"
        elif max_features == 8000:
            max_f = "8k"
        elif max_features == 10000:
            max_f = "10k"

        # set tags for the experiment
        mlflow.set_tag(
            "mlflow.runName",
            f"{vectorizer_type}_{n_gram_name}_{model_name}_{max_f}",
        )
        # set particular experiment description
        mlflow.set_tag(
            "mlflow.note.content",
            f"{model_name} model with {vectorizer_type}, ngram_range={n_gram_name}, max_features={max_features}",
        )

        mlflow.set_tag("experiment_type", f"{experiment_name}")
        mlflow.set_tag("model", model_name)

        # add a description
        mlflow.set_tag(
            "description",
            f"{model_name} with {vectorizer_type}, ngram_range={n_gram_name}, max_features={max_features}",
        )

        # log vectorizer parameters
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", tuple(n_gram))
        mlflow.log_param("vectorizer_max_features", max_features)

        # log model hyper-parameters
        if model_params[model_name]:
            for i, j in model_params[model_name].items():
                mlflow.log_param(i, j)

        model_signature = infer_signature(X_train_vect, y_train)

        # train the model
        y_pred, model = train_model(
            X_train_vect,
            y_train,
            X_test_vect,
            params=model_params[model_name],
            model_name=model_name,
        )
        infologger.info(f"{model_name} model trained successfully")

        # log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", round(accuracy, 3))

        # log classification report
        classification_rep = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # log confusion matrix
        cm_path = conf_matrix(y_test, y_pred, path=path)
        mlflow.log_artifact(cm_path, "confusion_matrix")

        # save model to local
        joblib.dump(
            model,
            f"{model_dir}/{model_name}_{vectorizer_type}_{n_gram_name}_{max_f}.joblib",
        )
        infologger.info(f"model saved successfully, path: {model_dir}")

        # log the model
        mlflow.sklearn.log_model(
            model,
            f"{model_name}_{vectorizer_type}_{n_gram_name}_{max_f}",
            signature=model_signature,
        )
        infologger.info("model successfully logged under mlflow")


def feature_extraction(
    vectorizer_type: str,
    max_features: int,
    ngram_range: Tuple,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    vectorizer_path: str,
) -> Tuple:

    if ngram_range == [1, 1]:
        n_gram_name = "unigram"
    elif ngram_range == [1, 2]:
        n_gram_name = "bigram"
    elif ngram_range == [1, 3]:
        n_gram_name = "trigram"

    if vectorizer_type == "bow":
        try:
            cv = CountVectorizer(
                max_features=max_features, ngram_range=tuple(ngram_range)
            )
            X_train_vect = cv.fit_transform(X_train)
            X_test_vect = cv.transform(X_test)
            infologger.info(f"input features vectorized (bow) successfully...")
        except Exception as e:
            infologger.error(
                f"some issue in bow, check feature_extraction() for issue. exception: {e}"
            )
        else:
            joblib.dump(
                cv, f"{vectorizer_path}/bow/bow_{n_gram_name}_{max_features}.joblib"
            )
            infologger.info(
                f"bow_vectorizer saved successfully, path: {vectorizer_path}/bow/bow_{n_gram_name}_{max_features}.joblib"
            )
            return X_train_vect, X_test_vect
    elif vectorizer_type == "tfidf":
        try:
            tfidf = TfidfVectorizer(
                max_features=max_features, ngram_range=tuple(ngram_range)
            )
            X_train_vect = tfidf.fit_transform(X_train)
            X_test_vect = tfidf.transform(X_test)
            infologger.info(f"input features vectorized (tfidf) successfully...")
        except Exception as e:
            infologger.error(
                f"some issue in tfidf, check feature_extraction() for issue. exception: {e}"
            )
        else:
            joblib.dump(
                tfidf,
                f"{vectorizer_path}/tfidf/tfidf_{n_gram_name}_{max_features}.joblib",
            )
            infologger.info(
                f"tfidf_vectorizer saved successfully, path: {vectorizer_path}/tfidf/tfidf_{n_gram_name}_{max_features}.joblib"
            )
            return X_train_vect, X_test_vect


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict,
    model_name: str,
) -> pd.Series:
    # train multiple models
    if model_name == "random_forest":
        try:
            model = RandomForestClassifier(**params)
            model.fit(x_train, y_train)
        except Exception as e:
            infologger.critical(
                f"unable to train random_forest model, check train_model() for issue. exception: {e}"
            )
        else:
            y_pred = model.predict(x_test)
            infologger.info("random_forest model trained successfully")
            return y_pred, model
    elif model_name == "gradient_boost":
        try:
            model = GradientBoostingClassifier(**params)
            model.fit(x_train, y_train)
        except Exception as e:
            infologger.critical(
                f"unable to train gradient_boost model, check train_model() for issue. exception: {e}"
            )
        else:
            y_pred = model.predict(x_test)
            infologger.info("gradient_boost model trained successfully")
            return y_pred, model
    elif model_name == "xgb":
        try:
            model = xgb.XGBClassifier(**params, num_class=3)
            model.fit(x_train, y_train)
        except Exception as e:
            infologger.critical(
                f"unable to train xgboost model, check train_model() for issue. exception: {e}"
            )
        else:
            y_pred = model.predict(x_test)
            infologger.info("xgboost model trained successfully")
            return y_pred, model
    elif model_name == "lgbm":
        try:
            model = lgb.LGBMClassifier(num_class=3)
            x_train = x_train.astype(float)
            x_test = x_test.astype(float)
            model.fit(x_train, y_train)
        except Exception as e:
            infologger.critical(
                f"unable to train lightgbm model, check train_model() for issue. exception: {e}"
            )
        else:
            y_pred = model.predict(x_test)
            infologger.info("lightgbm model trained successfully")
            return y_pred, model
    elif model_name == "catboost":
        try:
            model = CatBoostClassifier(**params)
            x_train = x_train.astype(float)
            x_test = x_test.astype(float)
            model.fit(x_train, y_train)
        except Exception as e:
            infologger.critical(
                f"unable to train catboost model, check train_model() for issue. exception: {e}"
            )
        else:
            y_pred = model.predict(x_test)
            infologger.info("catboost model trained successfully")
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

    curr_time = datetime.now().strftime("%d%m%y-%H%M%S")
    pathlib.Path.mkdir(
        pathlib.Path(f"{home_dir}/models/Exp-3/{curr_time}"),
        parents=True,
        exist_ok=True,
    )  # __make_change_here__
    model_dir = f"{home_dir}/models/Exp-3/{curr_time}"  # __make_change_here__

    for model_name in params["train_model"]["model_name"]:
        for vectorizer_type in params["build_features"]["vectorizer_type"]:
            for n_gram in params["build_features"]["n_gram"]:
                for max_features in params["build_features"]["max_features"]:
                    run_experiment(
                        df=df,
                        max_features=max_features,
                        model_name=model_name,
                        experiment_name=params["train_model"]["experiment_name"],
                        experiment_description=params["train_model"][
                            "experiment_description"
                        ],
                        vectorizer_type=vectorizer_type,
                        n_gram=n_gram,
                        model_params=params["train_model"]["hyperparams"],
                        path=f"{home_dir}/figures",
                        test_size=params["train_model"]["test_size"],
                        model_dir=f"{model_dir}",
                        vectorizer_path=f"{home_dir}/vectorizer",
                    )


if __name__ == "__main__":
    infologger.info("train_model.py as __main__")
    main()


# implement script to fetch the registered model from dagshub - pending
# some problem in exp 3, only few models saved in local - fix this issue (create directory and put model in it) - done
# fine tune lgbm - done
# create tune_model.py - done
# cover lightgbm and catboost campusx - lightgbm done
# use optuna - done
