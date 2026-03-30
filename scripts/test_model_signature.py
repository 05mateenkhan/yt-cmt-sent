import mlflow
import pytest
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri("https://dagshub.com/05mateenkhan/comments-analyzer.mlflow")


@pytest.mark.parametrize("model_name, stage", [
    ("yt_chrome_plugin_model", "Staging"),
])
def test_model_pipeline(model_name, stage):
    client = MlflowClient()

    # Get latest model version
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in '{stage}' for '{model_name}'"

    try:
        # Load model
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        # ✅ Correct input format (1D list of strings)
        input_data = ["hi how are you", "this is bad", "awesome video"]

        # Predict
        predictions = model.predict(input_data)

        # Assertions
        assert len(predictions) == len(input_data), "Prediction count mismatch"
        assert predictions is not None, "Predictions are None"

        print(f"Model '{model_name}' (stage: {stage}) works correctly.")

    except Exception as e:
        pytest.fail(f"Model test failed: {e}")