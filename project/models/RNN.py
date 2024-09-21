
from tensorflow.keras import models, layers

def initiate_compile_model():
    model = models.Sequential()
    model.add(layers.Dense(10, activation='relu', input_dim=3)) # highly recommend this option
    model.add(layers.Dense(7, activation='relu'))                               # instead of input_dim = 13
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
    loss = 'mse',
    optimizer = 'adam',
    metrics = ['mae'])
    return model
