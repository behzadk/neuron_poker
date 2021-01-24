from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
import numpy as np
import json

class Model:
    def __init__(self, name='equity_v1', load_model=None, model_dir=None,
        env=None, observation_shape=None, learning_rate=1e-3, layer_size=512):
        self.name = name
        self.observation_shape = observation_shape
        self.initiate_network(observation_shape, layer_size)
        self.model_dir = model_dir

        if load_model:
            self.load()

        self.model.compile(Adam(lr=learning_rate), metrics=['mse', tf.keras.metrics.MeanSquaredError()], loss='mse')

    def initiate_network(self, observation_shape, layer_size):
        tf.compat.v1.disable_eager_execution()

        self.model = Sequential()
        self.model.add(Dense(layer_size, activation='relu', input_shape=observation_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(layer_size, activation='relu'))
        self.model.add(Dropout(0.2))

        # Single output refers to the equity
        self.model.add(Dense(1, activation='linear'))


    def load(self):
        with open('{}/nn_{}_json.json'.format(self.model_dir, self.name), 'r') as architecture_json:
            model_json = json.load(architecture_json)

        self.model = model_from_json(model_json)
        self.model.load_weights('{}/nn_{}_weights.h5'.format(self.model_dir, self.name))

    def train(self, X, Y, num_epochs=150, batch_size=10):
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(self.name)
        tensorboard = TensorBoard(log_dir='{}/nn_equity_model/Graph/{}'.format(self.model_dir, timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)

        hist =self.model.fit(X, Y, epochs=num_epochs, shuffle=True,
                    batch_size=batch_size, callbacks=[tensorboard], verbose=1, validation_split=0.1)

        # Save the architecture
        model_json = self.model.to_json()
        with open("{}/nn_{}_json.json".format(self.model_dir, self.name), "w") as json_file:
            json.dump(model_json, json_file)

        # After training is done, we save the final weights.
        self.model.save_weights('{}/nn_{}_weights.h5'.format(self.model_dir, self.name), overwrite=True)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)