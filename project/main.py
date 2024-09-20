import json
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path

from project.gcp.helpers import load_data_from_bigquery
from project.preprocessors.helpers import haversine



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

        self.data = self.data[['fare_amount', 'passenger_count', 'is_day', 'distance_km']]
        return self

    def rescale(self):
        pass

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
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, self.outputs_path / "model.joblib")
        return self


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run-root-path", required=True)
    args = argparser.parse_args()
    trainer = Trainer(args.run_root_path)
    trainer.load_data()
    # trainer.load_data().preprocess().explore().split_data().train()
    print(trainer.data.head())
