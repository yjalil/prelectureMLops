import json
import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path
from tensorflow.keras import models, layers
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from project.gcp.helpers import load_data_from_bigquery
from project.feature_engineering.helpers import haversine, scaling
from project.models.RNN import initiate_compile_model


class Trainer():
    def __init__(self, run_root_path : str) -> None:
        self.run_root_path = Path(run_root_path)
        self.params_path = self.run_root_path / "params.json"
        self.outputs_path = self.run_root_path / "outputs"
        self.outputs_path.mkdir(exist_ok=True)

        with self.params_path.open("r") as f:
            self.params = json.load(f)

    def load_data(self):
        if self.params['local']:
            self.data = pd.read_csv(self.params["source"], nrows=self.params["source_limit"])
        else:
            self.data = load_data_from_bigquery(self.params["source"], self.params["source_limit"])

        return self

    def preprocess(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime'])
        self.data['is_day'] = self.data['pickup_datetime'].dt.hour.between(20, 7, inclusive='both').astype(int)
        self.data['distance_km'] = self.data.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'],
                                                    row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
        self.data['passenger_count'] = self.data['passenger_count'].replace(0, np.nan)
        self.data['distance'] = self.data['distance'].replace(0, np.nan)
        self.data= self.data[self.data['fare_amount']< 100]
        self.data= self.data[self.data['distance_km']< 40]
        self.data= self.data[self.data['distance_km']> 0]
        self.data= self.data[self.data['passenger_count']> 0]

        self.data = self.data[['fare_amount', 'passenger_count', 'is_day', 'distance_km']]
        return self

    def rescale(self):

        scaling(self.data)
        return self

    def explore(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data.corr(numeric_only=True), cmap="YlGnBu", annot=True)
        # sns.heatmap(self.data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(self.outputs_path / "correlation_matrix.png")
        return self

    def split_data(self):
        self.X = self.data[['distance_km', 'passenger_count', 'is_day']]
        self.y = self.data['fare_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.params['test_ratio'], random_state=42)
        return self

    def train(self):

        model = initiate_compile_model()
        es = EarlyStopping(patience =10,restore_best_weights = True)
        history = model.fit(self.X_train,self.y_train,validation_split = 0.3,
        shuffle = True,
        batch_size = 32,
        epochs = 100,
        callbacks = [es],
        verbose = 0
        )
        res = model.evaluate(self.X_test, self.y_test)[1]

        return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run-root-path", required=True)
    args = argparser.parse_args()
    trainer = Trainer(args.run_root_path)
    trainer.load_data()
    model = trainer.load_data().preprocess().rescale().explore().split_data().train()
    models.save_model(model, 'RNN')
    print( model.summary(), "model saved")
   # print(trainer.data.head())
