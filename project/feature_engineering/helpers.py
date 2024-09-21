import numpy as np
from sklearn.preprocessing import MinMaxScaler

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in kilometers
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in kilometers

def scaling(df):
    scaler = MinMaxScaler()
    df['passenger_count'] = scaler.fit_transform(df[['passenger_count']])
    df['distance_km'] = scaler.fit_transform(df[['distance_km']])
    return df
