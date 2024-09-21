from tensorflow.keras import callbacks
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
def RNNmodel(X_train_processed,y_train,X_test_processed,y_test):


    es = callbacks.EarlyStopping(patience=30, restore_best_weights=True)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1,activation='linear'))  # Sortie unique pour la régression (prédiction du prix)

    # Compilation du modèle
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entraînement du modèle
    history = model.fit(X_train_processed, y_train, callbacks=[es], epochs=500, batch_size=32, validation_split=0.3)

    # Évaluer le modèle sur les données de test
    test_loss, test_mae = model.evaluate(X_test_processed, y_test)
    print(f"Mean Absolute Error on test data: {test_mae}")

    # Prédictions sur les données de test
    y_pred = model.predict(X_test_processed)
    return (history,y_pred,test_mae,test_loss)
