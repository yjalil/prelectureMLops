import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    key = os.environ.get('KEY')
    gcp_project_id = os.environ.get('GCP_PROJECT_ID')
    gcp_bucket_id = os.environ.get('GCP_BUCKET_ID')

conf = Settings()
