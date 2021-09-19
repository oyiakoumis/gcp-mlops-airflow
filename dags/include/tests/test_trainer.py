from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from dags.include.trainer import ModelTrainer


@pytest.fixture()
@patch("dags.include.trainer.get_bucket")
def model_trainer(get_bucket):
    model_trainer = ModelTrainer("model_name", "bucket_name")
    get_bucket.assert_called_once_with("bucket_name")

    return model_trainer


def test_ModelTrainer_init(model_trainer):
    assert model_trainer.model_name == "model_name"


@patch("dags.include.trainer.BigQueryHook.get_pandas_df")
@patch("dags.include.trainer.BigQueryHook.__init__")
def test_load_data(bqh_init, get_pandas_df, model_trainer):
    bqh_init.return_value = None
    get_pandas_df.return_value = True
    assert model_trainer._load_data("query")
    bqh_init.assert_called_once_with(use_legacy_sql=False)
    get_pandas_df.assert_called_once_with(sql="query")


@patch("dags.include.trainer.pickle.loads")
def test_load_model(pkl_loads, model_trainer):
    model_trainer.bucket.blob.return_value.download_as_string.return_value = (
        "model_string"
    )
    pkl_loads.return_value = True

    assert model_trainer._load_model()

    model_trainer.bucket.blob.assert_called_once_with("model_name")
    pkl_loads.assert_called_once_with("model_string")


@patch("dags.include.trainer.pickle.dumps")
def test_save_model(pkl_dumps, model_trainer):
    pkl_dumps.return_value = "model_string"

    model_trainer._save_model("model")

    pkl_dumps.assert_called_once_with("model")
    model_trainer.bucket.blob.assert_called_once_with("model_name")
    model_trainer.bucket.blob.return_value.upload_from_string.assert_called_once_with(
        "model_string"
    )


@patch("dags.include.trainer.ModelTrainer._load_model")
@patch("dags.include.trainer.ModelTrainer._load_data")
@patch("dags.include.trainer.ModelTrainer._save_model")
def test_train_model(_save_model, _load_data, _load_model, model_trainer):
    _load_data.return_value = pd.DataFrame(
        {
            "col": [1],
            "date": [2],
            "quality": [3],
        }
    )

    model_trainer.train_model(templates_dict={"query": "sql"})

    _load_model.assert_called_once()
    _load_data.assert_called_once_with("sql")
    _load_model.return_value.fit.assert_called_once()
    _save_model.assert_called_once_with(_load_model.return_value)

    args, _ = _load_model.return_value.fit.call_args
    X, y = args

    assert_frame_equal(X, pd.DataFrame({"col": [1]}))
    assert_series_equal(y, pd.Series([3], name="quality"))
