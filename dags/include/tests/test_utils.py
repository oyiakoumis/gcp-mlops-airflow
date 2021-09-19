from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dags.include import utils


@patch("pandas.DataFrame.to_csv")
@patch("pandas.read_csv")
def test_clean_data_when_df_not_empty(read_csv, to_csv):
    df = pd.DataFrame(
        {
            "col 1": ["dummy"],
            "col 2": ["dummy"],
        }
    )

    read_csv.return_value = df

    utils.clean_data("gcs_data_path_in", "gcs_data_path_out", ds="date")

    assert_frame_equal(
        df, pd.DataFrame({"col_1": ["dummy"], "col_2": ["dummy"], "date": ["date"]})
    )

    read_csv.assert_called_once_with("gcs_data_path_in", sep=";")
    to_csv.assert_called_once_with("gcs_data_path_out", sep=";", index=False)


@patch("pandas.read_csv")
def test_clean_data_when_df_not_empty(read_csv):
    read_csv.return_value = pd.DataFrame()
    with pytest.raises(AssertionError):
        utils.clean_data("gcs_data_path_in", "gcs_data_path_out", ds="date")


def test_check_if_model_already_exists_when_model_exists():
    xcom_pull = MagicMock(return_value="model1")
    ti = MagicMock(xcom_pull=xcom_pull)
    assert utils.check_if_model_already_exists(ti=ti) == "dont_create_model_dummy"
    xcom_pull.assert_called_once_with(task_ids="bash_ml_engine_models_list")


@pytest.mark.parametrize("ml_engine_models_list", ["Listed 0 items.", ""])
def test_check_if_model_already_exists_when_model_does_not_exist(ml_engine_models_list):
    xcom_pull = MagicMock(return_value=ml_engine_models_list)
    ti = MagicMock(xcom_pull=xcom_pull)
    assert utils.check_if_model_already_exists(ti=ti) == "ml_engine_create_model"
    xcom_pull.assert_called_once_with(task_ids="bash_ml_engine_models_list")


@patch("dags.include.utils.Variable")
def test_set_current_version_name(Variable):
    Variable.get.return_value = "v1"
    utils.set_current_version_name()
    Variable.get.assert_called_once_with("CURRENT_VERSION_NAME")
    Variable.set.assert_called_once_with("CURRENT_VERSION_NAME", "v2")


@patch("dags.include.utils.storage.Client.get_bucket")
@patch("dags.include.utils.storage.Client.__init__", lambda self: None)
def test_get_bucket(get_bucket):
    get_bucket.return_value = True
    assert utils.get_bucket("bucket_name")
    get_bucket.assert_called_once_with("bucket_name")
