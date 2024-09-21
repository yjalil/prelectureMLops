from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from project.config import conf



def load_data_from_bigquery(source : str, limit : int = 100 ) -> pd.DataFrame:
    credentials = service_account.Credentials.from_service_account_file(conf.key)

    project_id = 'da1774'
    client = bigquery.Client(credentials= credentials,project=project_id)
    query_job = client.query(f'''SELECT * FROM `da1774.taxifare.{source}` LIMIT {limit} ''')
    results = query_job.result()
    return results.to_dataframe()
