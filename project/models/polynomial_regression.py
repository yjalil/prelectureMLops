import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import joblib
import numpy as np

class PolynomialRegressionModel:
    def __init__(self, X_train, X_test, y_train, y_test, outputs_path):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.outputs_path = outputs_path
        self.model = None

    def create_polynomial_pipeline(self, degree=2):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')


        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, ['distance_km']),
                ('cat', categorical_transformer, ['passenger_count', 'is_day'])
            ]
        )

        polynomial_features = PolynomialFeatures(degree=degree)
        model = LinearRegression()

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('poly_features', polynomial_features),
            ('model', model)
        ])

        return pipeline




    # validation croisée :
    def evaluate_model(self, X, y,degree, cv=10):
    # Créer un objet KFold pour la validation croisée
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    # Créer le pipeline complet
        pipeline = self.create_polynomial_pipeline(degree=degree)
    # Effectuer la validation croisée et calculer les scores
        scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

    # Calculer le RMSE moyen
        rmse_scores = np.sqrt(-scores)
        mean_rmse = np.mean(rmse_scores)

        return mean_rmse



    def train_and_evaluate(self, degrees_to_test):
        results = {}

        for degree in degrees_to_test:
            pipeline = self.create_polynomial_pipeline(degree=degree)
            pipeline.fit(self.X_train, self.y_train)

            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(self.y_test, y_pred)

            results[degree] = {
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
            }

            # Enregistrer le modèle pour chaque degré (facultatif)
            joblib.dump(pipeline, self.outputs_path / f"model_degree_{degree}.joblib")

        return results

   # def train(self):
    #    degrees_to_test = [1, 2, 3]
      #  results = self.train_and_evaluate(degrees_to_test)

        # Afficher les résultats pour chaque degré testé
      #  for degree, metrics in results.items():
      #      print(f"Degré: {degree}, MSE: {metrics['MSE']}, RMSE: {metrics['RMSE']}, R²: {metrics['R²']}")
    def train(self):
        degrees_to_test = [1, 2, 3]
        best_rmse = float('inf')
        best_degree = None
        best_pipeline = None

        for degree in degrees_to_test:
            self.degree = degree  # Stocker le degré courant
            pipeline = self.create_polynomial_pipeline(degree=degree)
            pipeline.fit(self.X_train, self.y_train)

        # Évaluer le modèle avec la validation croisée
            mean_rmse = self.evaluate_model(self.X_train, self.y_train,degree)

            print(f"Degré: {degree}, RMSE moyen: {mean_rmse:.4f}")

        # Vérifiez si ce modèle est le meilleur jusqu'à présent
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_degree = degree
                best_pipeline = pipeline

    # Enregistrer le meilleur modèle
        joblib.dump(best_pipeline, self.outputs_path / f"model_degree_{best_degree}.joblib")
        print(f"Meilleur modèle enregistré pour le degré {best_degree} avec un RMSE moyen de {best_rmse:.4f}.")
