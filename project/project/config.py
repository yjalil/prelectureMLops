import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    key = os.environ.get('KEY')

conf = Settings()

if __name__ == '__main__':
    print(conf.key)
