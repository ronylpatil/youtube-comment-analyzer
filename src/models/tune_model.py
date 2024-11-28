import yaml
import optuna  # type: ignore
import joblib
import mlflow
import dagshub  # type: ignore
import pathlib
import lightgbm as lgbm  # type: ignore
from functools import partial
from datetime import datetime
from src.logger import infologger
from sklearn.metrics import accuracy_score
from src.features.build_features import loadData
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from src.visualization.visualize import conf_matrix
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from src.models.train_model import feature_extraction

infologger.info("*** Executing: tune_model.py ***")


def objective_lgbm(trial, x_train, y_train) -> float:
    # Suggest values for the hyperparameters
    # boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    # data_sample_strategy = trial.suggest_categorical(
    # "data_sample_strategy", ["bagging", "goss"]
    # )
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    learning_rate = trial.suggest_float("learning_rate", 3e-3, 1e-1)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    is_unbalance = trial.suggest_categorical("is_unbalance", [True])
    class_weight = trial.suggest_categorical("class_weight", ["balanced"])

    # top_rate = trial.suggest_float("top_rate", 0.2, 0.4, step=0.1)
    # other_rate = trial.suggest_float("other_rate", 0.1, 0.4, step=0.1)
    # max_conflict_rate = trial.suggest_float("max_conflict_rate", 0.1, 0.5, step=0.1)

    # Create the RandomForestClassifier with suggested hyperparameters
    model = lgbm.LGBMClassifier(
        objective="multiclass",
        # boosting_type=boosting_type,
        # data_sample_strategy=data_sample_strategy,
        # data_sample_strategy="goss",
        n_estimators=n_estimators,
        num_class=3,
        learning_rate=learning_rate,
        max_depth=max_depth,
        metric="multi_logloss",
        is_unbalance=is_unbalance,
        class_weight=class_weight,
        # top_rate=top_rate,
        # other_rate=other_rate,
        verbose=-1,
        # max_conflict_rate=max_conflict_rate,
    )

    # Perform 3-fold cross-validation and calculate accuracy
    score = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy").mean()

    return score  # Return the accuracy score for Optuna to maximize


def main() -> None:

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    dagshub.init(
        repo_owner=params["train_model"]["repo_owner"],
        repo_name=params["train_model"]["repo_name"],
        mlflow=True,
    )

    clean_df = loadData(f"{home_dir}/data/processed/clean_data.csv").dropna()
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        clean_df["clean_text"],
        clean_df["category"],
        test_size=params["train_model"]["test_size"],
        stratify=clean_df["category"],
        random_state=42,
    )

    # convert [-1,0,1] to [0,1,2]
    target_map = {-1: 0, 0: 1, 1: 2}
    y_train = y_train.map(target_map).astype(float)
    y_test = y_test.map(target_map).astype(float)
    infologger.info(f"target varible mapped successfully. mapping: {target_map}")

    if params["tune_model"]["ngram_range"] == [1, 1]:
        n_gram_name = "unigram"
    elif params["tune_model"]["ngram_range"] == [1, 2]:
        n_gram_name = "bigram"
    elif params["tune_model"]["ngram_range"] == [1, 3]:
        n_gram_name = "trigram"

    # convert text to vectors
    X_train_vect, X_test_vect, vect_obj_path = feature_extraction(
        X_train=X_train,
        X_test=X_test,
        vectorizer_type=params["tune_model"]["vectorizer_type"],
        ngram_range=params["tune_model"]["ngram_range"],
        max_features=params["tune_model"]["max_features"],
        vectorizer_path=f"{home_dir}/vectorizer",
    )
    X_train_vect = X_train_vect.astype(float)
    X_test_vect = X_test_vect.astype(float)
    infologger.info("input data vectorized successfully")

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )  # We aim to maximize accuracy
    study.optimize(
        partial(objective_lgbm, x_train=X_train_vect, y_train=y_train),
        n_trials=params["tune_model"]["n_trials"],
    )

    # training model with optimized hyperparameters
    best_model = lgbm.LGBMClassifier(
        **study.best_trial.params,
        # objective="multiclass",
        # num_class=3,
        # class_weight="balanced",
        # metric="multi_logloss",
        # is_unbalance=True,
    )
    best_model.fit(X_train_vect, y_train)
    y_pred = best_model.predict(X_test_vect)

    curr_time = datetime.now().strftime("%d%m%y-%H%M%S")
    pathlib.Path.mkdir(
        pathlib.Path(f"{home_dir}/models/fine_tunned/{curr_time}"),
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        best_model,
        filename=f"{home_dir}/models/fine_tunned/{curr_time}/tunned_model.joblib",
    )

    # set mlflow experiment name
    mlflow.set_experiment(params["tune_model"]["experiment_name"])
    experiment_description = params["tune_model"]["experiment_description"]
    mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

    with mlflow.start_run() as run:
        # set tags for the experiment
        mlflow.set_tag(
            "mlflow.runName",
            "tunned_model_lightgbm",
        )
        # set particular experiment description
        mlflow.set_tag(
            "mlflow.note.content",
            f"lightgbm model with bow, ngram_range={n_gram_name}, max_features={params['tune_model']['max_features']}, n_trials={params['tune_model']['n_trials']}",
        )

        # experiment ta'
        mlflow.set_tag("experiment_type", params["tune_model"]["experiment_name"])
        mlflow.set_tag("model", "lightgbm")
        mlflow.set_tag("optimizer", "optuna")
        mlflow.set_tag("n_trials", params["tune_model"]["n_trials"])

        # log vectorizer parameters
        mlflow.log_param("vectorizer_type", params["tune_model"]["vectorizer_type"])
        mlflow.log_param("ngram_range", n_gram_name)
        mlflow.log_param(
            "vectorizer_max_features", params["tune_model"]["max_features"]
        )
        mlflow.log_params(study.best_trial.params)

        model_signature = infer_signature(X_train_vect, y_train)

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
        cm_path = conf_matrix(y_test, y_pred, path=f"{home_dir}/figures")
        mlflow.log_artifact(cm_path, "confusion_matrix")

        # log vectorizer object
        mlflow.log_artifact(vect_obj_path, artifact_path="vectorizer")

        # log the model
        mlflow.sklearn.log_model(
            best_model,
            f"best_model_optuna",
            signature=model_signature,
        )
        infologger.info("model successfully logged under mlflow")


if __name__ == "__main__":
    infologger.info("tune_model.py as __main__")
    main()

# log the vectorizer as well
