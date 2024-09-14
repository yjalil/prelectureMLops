import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    secret = os.environ.get('SECRET')

conf = Settings()

if __name__ == '__main__':
    print(conf.secret)
