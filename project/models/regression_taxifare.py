from project.config import conf


def hello() -> str:
    """This function says hello, dont give it any argument"""
    return "Hello, World!"

def get_secret() -> str:
    """This function returns a secret"""
    return conf.secret

def categorize_time(row):
    hour = row['pickup_datetime'].hour
    # Check if the hour is between 20 (8 PM) and 7 (7 AM)
    if hour >= 20 or hour < 7:
        return 1  # Night
    else:
        return 0  # Not Night

#df['is_night'] = df.apply(categorize_time, axis=1)


if __name__ == "__main__":
    print(hello())
    print(get_secret())
