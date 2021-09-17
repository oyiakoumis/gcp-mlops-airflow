import os
import datetime

from airflow.models import DAG, Variable
from airflow.contrib.sensors.gcs_sensor import GoogleCloudStorageObjectSensor
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from airflow.contrib.operators.gcs_to_gcs import (
    GoogleCloudStorageToGoogleCloudStorageOperator,
)
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.mlengine_operator import (
    MLEngineModelOperator,
    MLEngineVersionOperator,
)

from include import utils
from include.trainer import ModelTrainer
from include.queries import get_train_data

GCS_BASE_PATH = "gs://{{ var.value.PROJECT_BUCKET }}/"
TRAIN_OBJECT = f"{Variable.get('DATA_FOLDER')}/{Variable.get('TRAIN_NAME')}"
TRAIN_OBJECT_CLEANED = (
    f"{Variable.get('DATA_FOLDER')}/cleaned_{Variable.get('TRAIN_NAME')}"
)
BQ_TABLE = f"{Variable.get('PROJECT_ID')}:{Variable.get('DATASET')}.{Variable.get('TABLE_NAME')}"

default_args = {
    "start_date": datetime.datetime(2021, 9, 13),
}

with DAG(
    "wine_quality",
    schedule_interval=datetime.timedelta(days=14),
    default_args=default_args,
    catchup=False,
) as dag:
    check_if_train_data_exists = GoogleCloudStorageObjectSensor(
        task_id="check_if_train_data_exists",
        bucket=Variable.get("PROJECT_BUCKET"),
        object=TRAIN_OBJECT,
        poke_interval=60,
        timeout=300,
        mode="reschedule",
    )

    # archive old model
    archive_old_model = GoogleCloudStorageToGoogleCloudStorageOperator(
        task_id="archive_old_model",
        source_bucket=Variable.get("PROJECT_BUCKET"),
        source_object="model.pkl",
        destination_bucket=Variable.get("PROJECT_BUCKET"),
        destination_object=os.path.join("archive", "models", "model_{{ ds }}.pkl"),
    )

    # Remove space from columns and add timestamp
    gcs_data_path_in = GCS_BASE_PATH + f"{TRAIN_OBJECT}"
    gcs_data_path_out = GCS_BASE_PATH + f"{TRAIN_OBJECT_CLEANED}"

    print(gcs_data_path_in)

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=utils.clean_data,
        op_args=(
            gcs_data_path_in,
            gcs_data_path_out,
        ),
        provide_context=True,
    )

    # Add cleaned train data to BigQuery table
    gcs_to_bq = GoogleCloudStorageToBigQueryOperator(
        task_id="write_train_data_to_bq",
        bucket=Variable.get("PROJECT_BUCKET"),
        source_objects=[TRAIN_OBJECT_CLEANED],
        field_delimiter=";",
        destination_project_dataset_table=BQ_TABLE,
        create_disposition="CREATE_IF_NEEDED",
        write_disposition=Variable.get("WRITE_DISPOSITION"),  # WRITE_APPEND in prod
        schema_object="{{ var.value.DATA_FOLDER }}/train_schema.json",
        skip_leading_rows=1,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=ModelTrainer(
            "model.pkl", Variable.get("PROJECT_BUCKET")
        ).train_model,
        templates_dict={"query": get_train_data},
        provide_context=True,
    )

    # List currently existing models on AI Platform and pass to the next task
    # via the use of an XCom.
    bash_ml_engine_models_list = BashOperator(
        task_id="bash_ml_engine_models_list",
        xcom_push=True,
        bash_command="gcloud ai-platform models list --region=global --filter='name:{}'".format(
            Variable.get("MODEL_NAME")
        ),
    )

    check_if_model_exists = BranchPythonOperator(
        task_id="check_if_model_already_exists",
        python_callable=utils.check_if_model_already_exists,
        provide_context=True,
    )

    # In case the model doesn't exist, using an MLEngineModelOperator to create the new model.
    ml_engine_create_model = MLEngineModelOperator(
        task_id="ml_engine_create_model",
        project_id=Variable.get("PROJECT_ID"),
        model={"name": Variable.get("MODEL_NAME")},
        operation="create",
    )

    # NoOp in the case that the model already exists.
    dont_create_model_dummy = DummyOperator(
        task_id="dont_create_model_dummy",
    )

    set_current_version_name = PythonOperator(
        task_id="set_current_version_name",
        python_callable=utils.set_current_version_name,
        trigger_rule="none_failed_or_skipped",
    )

    # MLEngineVersionOperator with operation set to "create" to create a new
    # version of our model
    ml_engine_create_version = MLEngineVersionOperator(
        task_id="ml_engine_create_version",
        project_id=Variable.get("PROJECT_ID"),
        model_name=Variable.get("MODEL_NAME"),
        version_name=Variable.get("CURRENT_VERSION_NAME"),
        version={
            "name": Variable.get("CURRENT_VERSION_NAME"),
            "deploymentUri": f"gs://{Variable.get('PROJECT_BUCKET')}",
            "runtimeVersion": "2.6",
            "framework": "scikit-learn",
            "pythonVersion": "3.7",
        },
        operation="create",
    )

    # MLEngineVersionOperator with operation set to "set_default" to set our
    # newly deployed version to be the default version.
    ml_engine_set_default_version = MLEngineVersionOperator(
        task_id="ml_engine_set_default_version",
        project_id=Variable.get("PROJECT_ID"),
        model_name=Variable.get("MODEL_NAME"),
        version_name=Variable.get("CURRENT_VERSION_NAME"),
        version={"name": Variable.get("CURRENT_VERSION_NAME")},
        operation="set_default",
    )

    (
        check_if_train_data_exists
        >> [archive_old_model, clean_data]
        >> gcs_to_bq
        >> train_model
        >> bash_ml_engine_models_list
        >> check_if_model_exists
        >> [ml_engine_create_model, dont_create_model_dummy]
        >> set_current_version_name
        >> ml_engine_create_version
        >> ml_engine_set_default_version
    )
