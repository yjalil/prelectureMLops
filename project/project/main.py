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


from project.gcp.helpers import load_data_from_bigquery
from project.feature_engineering.helpers import haversine, scaling
from project.models.RNN import initiate_compile_model


class Trainer():
    def __init__(self, run_root_path : str) -> None:
        self.run_root_path = Path(run_root_path)
        #print("run root path: ", self.run_root_path)
        self.params_path = self.run_root_path / "params.json"
        #print("params path: ", self.params_path)
        self.outputs_path = self.run_root_path / "outputs"
        #self.outputs_path.mkdir(exist_ok=True)
        #print("output path: ", self.outputs_path)
        with self.params_path.open("r") as f:
            self.params = json.load(f)

    def load_data(self):
        if self.params['local']:
            self.data = pd.read_csv(self.params["source"], nrows=self.params["source_limit"])
            #self.data = pd.read_csv(self.params["source"])
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
        self.data['distance_km'] = self.data['distance_km'].replace(0, np.nan)
        self.data= self.data[self.data['fare_amount']< 100]
        self.data= self.data[self.data['distance_km']< 40]
        self.data= self.data[self.data['distance_km']> 0]
        self.data= self.data[self.data['passenger_count']> 0]

        self.data = self.data[['fare_amount', 'passenger_count', 'is_day', 'distance_km']]
        print(" data shape", self.data.shape)
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
        self.res = model.evaluate(self.X_test, self.y_test)[1]
        self.model = model
        self.history = history
        return self

    def plot_history(self):
        # Setting figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))
        # Create the plots
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax2.plot(self.history.history['mae'])
        ax2.plot(self.history.history['val_mae'])
        # Set titles and labels
        ax1.set_title('Model loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax2.set_title('MAE')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        # Set limits for y-axes
        ax1.set_ylim(ymin=0, ymax=200)
        ax2.set_ylim(ymin=0, ymax=25)
        # Generate legends
        ax1.legend(['Train', 'Validation'], loc='best')
        ax2.legend(['Train', 'Validation'], loc='best')
        # Show grids
        ax1.grid(axis="x", linewidth=0.5)
        ax1.grid(axis="y", linewidth=0.5)
        ax2.grid(axis="x", linewidth=0.5)
        ax2.grid(axis="y", linewidth=0.5)

        plt.savefig(self.outputs_path /"history_loss_mae.png")
        return self

    def save_model(self):
        models.save_model(self.model, self.outputs_path/"results.keras")
        return self


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run-root-path", required=True)
    args = argparser.parse_args()
    #print("args", args.run_root_path)
    trainer = Trainer(args.run_root_path)
    trainer.load_data()
    model = trainer.load_data().preprocess().rescale().explore().split_data().train().save_model()
    model = model.plot_history()
    print( f"model evaluation, {trainer.res}")
   # print(trainer.data.head())
