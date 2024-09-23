from google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account
import pandas as pd
import json
from project.config import conf



def load_data_from_bigquery(source : str, limit : int = 100 ) -> pd.DataFrame:
    credentials = service_account.Credentials.from_service_account_file(conf.key)
    client = bigquery.Client(credentials= credentials,project=conf.gcp_project_id)
    query_job = client.query(f'''SELECT * FROM `{conf.gcp_project_id}.taxifare.{source}` LIMIT {limit} ''')
    results = query_job.result()
    return results.to_dataframe()

def load_params_from_buckets(folder : str) -> dict:
    credentials = service_account.Credentials.from_service_account_file(conf.key)
    storage_client = storage.Client(credentials=credentials, project=conf.gcp_project_id)
    bucket = storage_client.bucket(conf.gcp_bucket_id)
    blob = bucket.blob(f"runs/{folder}/params.json")
    return json.loads(blob.download_as_string())

def save_buffer_to_bucket(folder : str, buffer : bytes, file_path : str) -> None:
    credentials = service_account.Credentials.from_service_account_file(conf.key)
    storage_client = storage.Client(credentials=credentials, project=conf.gcp_project_id)
    bucket = storage_client.bucket(conf.gcp_bucket_id)
    blob = bucket.blob(f"runs/{folder}/{file_path}")
    blob.upload_from_string(buffer)
