from project.config import conf


def hello() -> str:
    """This function says hello, dont give it any argument"""
    return "Hello, World!"

def get_secret() -> str:
    """This function returns a secret"""
    return conf.secret


