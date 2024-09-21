from project.config import conf
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, cross_val_score

class LinearRegressionModel:
    def __init__(self, df: pd.DataFrame):
            self.df = df
            self.pipeline = None
    def hello() -> str:
        """This function says hello, dont give it any argument"""
        return "Hello, World!"

    def get_secret() -> str:
        """This function returns a secret"""
        return conf.secret

    #def categorize_time(self,row):
    #    hour = row['pickup_datetime'].hour
        # Check if the hour is between 20 (8 PM) and 7 (7 AM)
    #    if hour >= 20 or hour < 7:
    #        return 1  # Night
    #    else:
      #     return 0  # Not Night

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

    def create_linear_pipeline(self) -> Pipeline:
            """Creates a linear regression pipeline."""
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    def create_linear_pipeline(self) -> Pipeline:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, ['distance_km']),
                ('cat', categorical_transformer, ['passenger_count', 'is_day'])
            ]
        )

        model = LinearRegression()

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline
    def train_and_evaluate(self, X, y):
        """Trains and evaluates a linear regression model."""

        # Créer et ajuster le modèle de régression linéaire
        self.pipeline = self.create_linear_pipeline()

        # Ajuster le modèle
        self.pipeline.fit(X, y)

        # Faire des prédictions
        y_pred = self.pipeline.predict(X)

        # Calculer les métriques de performance
        mse = mean_squared_error(y, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y, y_pred)

        results = {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }

        return results

# Calculate distance and create a new column in the DataFrame
#df['distance_km'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

#if __name__ == "__main__":
 #   print(hello())
 #   print(get_secret())
