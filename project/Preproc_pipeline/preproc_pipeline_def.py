from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def preproc_pipeline(filtered_df):

        # Prétraitement des données
        # Remplacer les valeurs de 0 par NaN pour les traiter
        filtered_df['passenger_count'] = filtered_df['passenger_count'].replace(0, np.nan)
        filtered_df['distance'] = filtered_df['distance'].replace(0, np.nan)

        # Pipeline de prétraitement
        preprocessing_pipeline = Pipeline(steps=[
            # Imputation des 0 dans 'passenger_count' par la valeur la plus fréquente
            ('impute_passenger_count', SimpleImputer(strategy='most_frequent')),

            # Imputation des 0 dans 'distance' par la médiane
            ('impute_distance', SimpleImputer(strategy='median')),

            # Mise à l'échelle des données avec MinMaxScaler
            ('scaler', MinMaxScaler())
        ])

        # Séparer les variables d'entrée (X) et la variable cible (y)
        X = filtered_df[['distance', 'passenger_count']]
        y = filtered_df['fare_amount']

        # Diviser en ensemble d'entraînement et de test (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Appliquer le pipeline de prétraitement
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        return (X_train_processed,X_test_processed)
