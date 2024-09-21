import json
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from models.polynomial_regression import PolynomialRegressionModel
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from project.models.regression_taxifare  import LinearRegressionModel  # Assurez-vous que ce chemin est correct
#from project.gcp.helpers import load_data_from_bigquery
#from project.models.regression_taxifare  import LinearRegressionModel.haversine



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
       # else:
        #    self.data = load_data_from_bigquery(self.params["source"], self.params["source_limit"])

        return self


    def preprocess(self):
        # Supprimer les valeurs manquantes et les doublons
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        #je remplance les nan par la mediane
        #self.data['fare_amount'].fillna(self.data['fare_amount'].mean(), inplace=True)
        # Convertir la colonne 'pickup_datetime' en datetime
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime'])

        # Créer la colonne 'is_day'
        self.data['is_day'] = (~self.data['pickup_datetime'].dt.hour.between(20, 7)).astype(int)  # 1 pour le jour, 0 pour la nuit

        # Calculer la distance en km
        self.data['distance_km'] = self.data.apply(lambda row: LinearRegressionModel.haversine(row['pickup_longitude'], row['pickup_latitude'],
                                                    row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

        # Supprimer les outliers basés sur 'fare_amount'
        Q1 = self.data['fare_amount'].quantile(0.25)
        Q3 = self.data['fare_amount'].quantile(0.75)
        IQR = Q3 - Q1

        # Définir une limite pour les outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrer les données pour supprimer les outliers
        self.data = self.data[(self.data['fare_amount'] >= lower_bound) & (self.data['fare_amount'] <= upper_bound)]

        # Sélectionner uniquement les colonnes pertinentes
        self.data = self.data[['fare_amount', 'passenger_count', 'is_day', 'distance_km']]

        return self
    def rescale(self):
        scaler = StandardScaler()

        # Appliquer le scaler uniquement sur les colonnes numériques
        self.X_train[['distance_km', 'passenger_count']] = scaler.fit_transform(self.X_train[['distance_km', 'passenger_count']])
        self.X_test[['distance_km', 'passenger_count']] = scaler.transform(self.X_test[['distance_km', 'passenger_count']])

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
         # Vérifier les catégories uniques dans les données d'entraînement et de test
        print("Catégories uniques dans X_train:", self.X_train['is_day'].unique())
        print("Catégories uniques dans X_test:", self.X_test['is_day'].unique())

        return self


    def train_polynomial_model(self):
        model_instance = PolynomialRegressionModel(self.X_train, self.X_test, self.y_train, self.y_test, self.outputs_path)

       # Appeler la méthode train sur l'instance du modèle
        model_instance.train()


    def train_linear_model(self):
        linear_model_instance = LinearRegressionModel(self.X_train)  # Passer uniquement X_train ici

    # Appeler la méthode train_and_evaluate en passant X_test et y_test
        linear_results = linear_model_instance.train_and_evaluate(self.X_test, self.y_test)  # Passer X_test et y_test
        print("Linear Regression Results:", linear_results)


    def main(self):
        # Charger les données et exécuter les étapes de traitement et d'entraînement
        (self.load_data()
             .preprocess()
             .explore()
             .split_data()
             .rescale())

        # Entraîner le modèle de régression polynomiale
        self.train_polynomial_model()
        self.train_linear_model()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run-root-path", required=True)
    args = argparser.parse_args()
    trainer = Trainer(args.run_root_path)
    trainer.load_data()
    # trainer.load_data().preprocess().explore().split_data().train()
    print(trainer.data.head())
    trainer.main()
