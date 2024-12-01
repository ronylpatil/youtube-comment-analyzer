# do prediction using best model sitting in mlflow server
import json
import click
import joblib
import mlflow
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
    except Exception as e:
        print(
            f"unable to fetch model from mlflow server, check get_model.py >> get_model(...) for potential issue. exception: {e}"
        )
    else:
        return model, model_details


@click.command()
@click.argument("mlflow_uri")
@click.argument("registered_model_name")
@click.argument("alias")
def save_model(mlflow_uri, registered_model_name, alias) -> None:
    model, model_details = fetch_model(
        mlflow_uri=mlflow_uri, registered_model_name=registered_model_name, alias=alias
    )
    joblib.dump(model, "./model.joblib")

    with open("./model_details.json", "w") as jsn:
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

# solve model dir bug -> only 2 dir are tracked by DVC each folder should be tracked by DVC
