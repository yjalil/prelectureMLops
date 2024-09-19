from project.config import conf
import numpy as np

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
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in kilometers

# Calculate distance and create a new column in the DataFrame
#df['distance_km'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

if __name__ == "__main__":
    print(hello())
    print(get_secret())
