import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    key = os.environ.get('KEY')

conf = Settings()
