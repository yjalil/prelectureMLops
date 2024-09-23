import json
import io
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path

from project.gcp.helpers import load_data_from_bigquery
from project.gcp.helpers import load_params_from_buckets
from project.gcp.helpers import save_buffer_to_bucket
from project.preprocessors.helpers import haversine



class Trainer():
    def __init__(self, run_root_path : str) -> None:
        self.run_root_path = Path(run_root_path)
        self.params = load_params_from_buckets(self.run_root_path)
        self.outputs_path = self.run_root_path / "outputs"

    def load_data(self):
        if self.params['local']:
            self.data = pd.read_csv(self.params["source"], nrows=self.params["source_limit"])
        else:
            self.data = load_data_from_bigquery(self.params["source"], self.params["source_limit"])
        print("Loading data ...")

        return self

    def preprocess(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data['tpep_pickup_datetime'] = pd.to_datetime(self.data['tpep_pickup_datetime'])
        self.data['is_day'] = self.data['tpep_pickup_datetime'].dt.hour.between(20, 7, inclusive='both').astype(int)
        self.data = self.data[['fare_amount', 'passenger_count', 'is_day', 'trip_distance']]
        print("Preprocessing data ...")
        return self

    def rescale(self):
        pass

    # def explore(self):
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(self.data.corr(numeric_only=True), cmap="YlGnBu", annot=True)
    #     # sns.heatmap(self.data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    #     plt.title('Correlation Matrix')
    #     plt.savefig(self.outputs_path / "correlation_matrix.png")
    #     print("Exploring data correlations...")
    #     return self

    def split_data(self):
        self.X = self.data[['trip_distance', 'passenger_count', 'is_day']]
        self.y = self.data['fare_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.params['test_ratio'], random_state=42)
        print("Splitting data ...")
        return self


    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)

        save_buffer_to_bucket(self.outputs_path, buffer.getvalue(), "model.joblib")
        print("Training model ...")
        return self
    def eveluate(self):
        pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-run", required=True)
    args = argparser.parse_args()
    trainer = Trainer(args.run)
    trainer.load_data().preprocess().split_data().train()
    print(trainer.params)
