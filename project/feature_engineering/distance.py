
from project.feature_engineering.helpers import haversine

def distancedf(df):
    df["distance"]= df.apply(lambda x: haversine(x.pickup_longitude,x.pickup_latitude,x.dropoff_longitude,x.dropoff_latitude),axis=1)
    return df
