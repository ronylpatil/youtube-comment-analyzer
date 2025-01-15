# do prediction using best model sitting in mlflow server
import json
import click
import joblib
import mlflow
import pathlib
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient


def fetch_model(mlflow_uri, registered_model_name, alias) -> click.Tuple:
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        model_details = client.get_model_version_by_alias(
            name=registered_model_name, alias=alias
        )

        model = load_model(f"models:/{model_details.name}@{model_details.aliases[0]}")

        try:
            artifact_path = client.list_artifacts(
                run_id=model_details.run_id, path="vectorizer"
            )
        except Exception as e:
            print(
                f"vectorier is not logged in existing experiment, check artifacts for potential issue. exception: {e}"
            )
        else:
            vectorizer_path = client.download_artifacts(
                run_id=model_details.run_id, path=f"{artifact_path[0].path}"
            )
            vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print(
            f"unable to fetch model from mlflow server, check get_model.py >> fetch_model(...) for potential issue. exception: {e}"
        )
    else:
        return model, model_details, vectorizer


@click.command()
@click.argument("mlflow_uri")
@click.argument("registered_model_name")
@click.argument("alias")
def save_model(mlflow_uri, registered_model_name, alias) -> None:
    model, model_details, vectorizer = fetch_model(
        mlflow_uri=mlflow_uri, registered_model_name=registered_model_name, alias=alias
    )

    dir_path = pathlib.Path(f"{pathlib.Path(__file__).parent.as_posix()}/prod_model")         # running in local sys or CICD pipeline
    pathlib.Path.mkdir(
        dir_path,
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, f"{dir_path}/model.joblib")
    joblib.dump(vectorizer, f"{dir_path}/vectorizer.joblib")

    with open(f"{dir_path}/model_details.json", "w") as jsn:
        json.dump(
            {
                "name": model_details.name,
                "version": model_details.version,
                "alias": model_details.aliases[0],
                "runid": model_details.run_id,
            },
            jsn,
        )


if __name__ == "__main__":
    save_model()

# alias = "production"
# registered_model_name = "lightgbm_v1.0"
# mlflow_uri = "https://dagshub.com/ronylpatil/youtube-comment-analyzer.mlflow"
# cmd: python prod/get_model.py "https://dagshub.com/ronylpatil/youtube-comment-analyzer.mlflow" "lightgbm_v1.0" "production"
