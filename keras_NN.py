from keras.models import Sequential
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
import os
from keras.models import load_model

class KerasNeuralNetwork(object):

    def custom_accuracy(self, y_true, y_pred):
        return K.mean(tf.reduce_all(K.equal(y_true, K.round(y_pred)), axis=1))

    def __init__(self, input_size, output_size, hidden_layers, name, activation_output='sigmoid', load_model_path=''):

        self.model = Sequential()

        self.name = name

        # Create model checkpoints directory
        if not os.path.exists('Models/%s' % self.name):
            os.makedirs('Models/%s' % self.name)

        # Hidden Layers
        hidden_layers_sizes = np.round(np.linspace(input_size, output_size, hidden_layers + 1)).astype(int)

        self.model.add(Dense(units=hidden_layers_sizes[0], activation='relu', input_shape=(896,)))

        for i, hidden_layer_size in enumerate(hidden_layers_sizes[1:-1]):
            self.model.add(Dense(units=hidden_layer_size, activation='relu'))
            if i%2 == 0:
                self.model.add(Dropout(0.2))

        # Output Layer
        self.model.add(Dense(units=output_size, activation=activation_output))

        if load_model_path:
            self.model.load_weights(load_model_path)
        
        self.model.compile(loss=binary_crossentropy, optimizer='sgd',
                           metrics=[self.custom_accuracy])
        print self.model.summary()
        exit()
        tensorboard_callback = TensorBoard(log_dir=os.path.join('logs/', self.name), histogram_freq=1, batch_size=32,
                                           write_graph=False,
                                           write_grads=True)

        checkpoint_callback = ModelCheckpoint(filepath="Models/%s/model_{epoch:02d}_{val_loss:.2f}" % (self.name), monitor='val_loss', verbose=1,
                                              save_best_only=True, save_weights_only=False, mode="min", period=1)

        self.callbacks = [tensorboard_callback, checkpoint_callback]

    def fit_generator(self, training_generator, validation_data):
        self.model.fit_generator(generator=training_generator, steps_per_epoch=100000,
                                 epochs=1000000, verbose=1, callbacks=self.callbacks,
                                 validation_data=validation_data, validation_steps=50,
                                 use_multiprocessing=True)
