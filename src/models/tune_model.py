import yaml
import optuna  # type: ignore
import pathlib
import lightgbm as lgbm  # type: ignore
from functools import partial
from src.logger import infologger
from sklearn.metrics import accuracy_score
from src.features.build_features import loadData
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from src.visualization.visualize import conf_matrix
from sklearn.model_selection import train_test_split
from src.models.train_model import feature_extraction

infologger.info("*** Executing: tune_model.py ***")


def objective_lgbm(trial, x_train, y_train) -> float:
    # Suggest values for the hyperparameters
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    data_sample_strategy = trial.suggest_categorical(
        "data_sample_strategy", ["bagging", "goss"]
    )
    n_estimators = trial.suggest_int("n_estimators", 100, 800, step=20)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.3, step=0.01)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    top_rate = trial.suggest_float("top_rate", 0.2, 0.6, step=0.1)
    other_rate = trial.suggest_float("other_rate", 0.1, 0.5, step=0.1)
    # max_conflict_rate = trial.suggest_float("max_conflict_rate", 0.1, 0.5, step=0.1)

    # Create the RandomForestClassifier with suggested hyperparameters
    model = lgbm.LGBMClassifier(
        objective="multiclass",
        boosting_type=boosting_type,
        data_sample_strategy=data_sample_strategy,
        n_estimators=n_estimators,
        num_class=3,
        learning_rate=learning_rate,
        max_depth=max_depth,
        top_rate=top_rate,
        other_rate=other_rate,
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

    clean_df = loadData(f"{home_dir}/data/processed/clean_data.csv").dropna()
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        clean_df["clean_text"],
        clean_df["category"],
        test_size=params["train_model"]["test_size"],
    )
    
    # convert [-1,0,1] to [0,1,2]
    target_map = {-1: 0, 0: 1, 1: 2}
    y_train = y_train.map(target_map).astype(float)
    y_test = y_test.map(target_map).astype(float)
    infologger.info(f"target varible mapped successfully. mapping: {target_map}")

    # convert text to vectors
    X_train_vect, X_test_vect = feature_extraction(
        X_train=X_train,
        X_test=X_test,
        vectorizer_type="bow",
        ngram_range=(1, 2),
        max_features=10000,
    )
    X_train_vect = X_train_vect.astype(float)
    X_test_vect = X_test_vect.astype(float)
    infologger.info("input data vectorized successfully")

    # print accuracy and classification report

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )  # We aim to maximize accuracy
    study.optimize(
        partial(objective_lgbm, x_train=X_train_vect, y_train=y_train), n_trials=1
    )

    # training model with optimized hyperparameters
    best_model = lgbm.LGBMClassifier(**study.best_trial.params)
    best_model.fit(X_train_vect, y_train)
    y_pred = best_model.predict(X_test_vect)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}\n")

    classification_rep = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    print(f"Classification Report: \n{classification_rep}\n")

    cm_path = conf_matrix(y_test, y_pred, path=f"{home_dir}/figures")
    print(f"Confusion Matrix Path: {cm_path}")


if __name__ == "__main__":
    infologger.info("tune_model.py as __main__")
    main()
