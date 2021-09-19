import pickle
from airflow.contrib.hooks.bigquery_hook import BigQueryHook

from .utils import get_bucket


class ModelTrainer:
    def __init__(self, model_name: str, bucket_name: str):
        self.model_name = model_name
        self.bucket = get_bucket(bucket_name)

    @staticmethod
    def _load_data(query: str):
        """Load data from Google Cloud Storage and convert it to pd.DataFrame"""
        bq_hook = BigQueryHook(use_legacy_sql=False)
        return bq_hook.get_pandas_df(sql=query)

    def _load_model(self):
        """Load model from Google Cloud Storage"""
        # select bucket file
        blob = self.bucket.blob(self.model_name)

        # download blob into an in-memory file object
        model_string = blob.download_as_string()

        # load pickled model
        return pickle.loads(model_string)

    def _save_model(self, model):
        """Save model to Google Cloud Storage"""
        model_string = pickle.dumps(model)

        # Upload model to project folder
        blob = self.bucket.blob(self.model_name)
        blob.upload_from_string(model_string)

    def train_model(self, **kwargs):
        """
        Load, train sklearn model and save it to Google Cloud Storage
        """
        model = self._load_model()
        df = self._load_data(kwargs["templates_dict"]["query"])

        # Train model
        X, y = (
            df[[c for c in df.columns if c not in ["quality", "date"]]],
            df["quality"],
        )
        model.fit(X, y)

        self._save_model(model)
