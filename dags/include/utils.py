import pandas as pd
from airflow.models import Variable
from google.cloud import storage


def clean_data(gcs_data_path_in: str, gcs_data_path_out: str, **kwargs):
    """
    # Remove space from columns and add timestamp
    """
    df = pd.read_csv(gcs_data_path_in, sep=";")

    # Check if csv is empty
    assert len(df) > 0

    # BigQuery doesn't support column named with spaces
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # add timestamp
    df.loc[:, "date"] = kwargs["ds"]

    df.to_csv(gcs_data_path_out, sep=";", index=False)


def check_if_model_already_exists(**kwargs):
    """
    Python callable which returns the appropriate task based on if the model
    we want to deploy exists or not already.
    """
    ml_engine_models_list = kwargs["ti"].xcom_pull(
        task_ids="bash_ml_engine_models_list"
    )

    if len(ml_engine_models_list) == 0 or ml_engine_models_list == "Listed 0 items.":
        return "ml_engine_create_model"
    else:
        return "dont_create_model_dummy"


def set_current_version_name():
    current_version = int(Variable.get("CURRENT_VERSION_NAME")[1])
    Variable.set("CURRENT_VERSION_NAME", "v{}".format(current_version + 1))


def get_bucket(bucket_name):
    storage_client = storage.Client()
    return storage_client.get_bucket(bucket_name)
