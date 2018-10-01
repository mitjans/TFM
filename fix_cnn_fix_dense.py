from ChessParser import ChessParser
import numpy as np
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
import os

# Prepare gpu
# Disable Tensorflow environments about CPU instructions
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto() 
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config = config))

# Variables
training_data_dir = "Databases/Chess/TrainData"
validation_data_dir = "Databases/Chess/TestData"

# Hyperparameters
number_of_iterations = 1
input_shape = (8, 8, 6)

first_cnn_layer_kernel_shape = (8, 8)
first_cnn_layer_depth = 128
second_cnn_layer_kernel_shape = (8, 8)
second_cnn_layer_depth = 128
third_cnn_layer_kernel_shape = (6, 6)
third_cnn_layer_depth = 256
forth_cnn_layer_kernel_shape = (6, 6)
forth_cnn_layer_depth = 256
fifth_cnn_layer_kernel_shape = (4, 4)
fifth_cnn_layer_depth = 512
sixth_cnn_layer_kernel_shape = (4, 4)
sixth_cnn_layer_depth = 512
seventh_cnn_layer_kernel_shape = (2, 2)
seventh_cnn_layer_depth = 1024
cnn_padding = 'same'
cnn_activation = 'relu'

first_pool_size = 3
second_pool_size = 3
third_pool_size = 2

number_of_hidden_layers = 4
first_hidden_layer_size = 2048
second_hidden_layer_size = 1024
third_hidden_layer_size = 512
forth_hidden_layer_size = 32
dense_activation = 'relu'

name = '2d-6cnn-128-256-512-4hdl-2048-1024-512-32'

if not os.path.exists('Models/%s' % name):
    os.makedirs('Models/%s' % name)

model = Sequential()
model.add(Conv2D(128, (8, 8), padding='same', activation='relu', input_shape=(8, 8, 6),))
model.add(Conv2D(128, (8, 8), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=1))
model.add(Conv2D(256, (6, 6), padding='same', activation='relu'))
model.add(Conv2D(256, (6, 6), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=1))
model.add(Conv2D(512, (4, 4), padding='same', activation='relu'))
model.add(Conv2D(512, (4, 4), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(1024, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(1024, (2, 2), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(32, activation='sigmoid'))

def custom_accuracy(y_true, y_pred):
    return K.mean(tf.reduce_all(K.equal(y_true, K.round(y_pred)), axis=1))

model.compile(loss=binary_crossentropy, optimizer='sgd', metrics=[custom_accuracy])
print model.summary()
exit()
print "Generating validation data..."
validation_data_generator = ChessParser.cnn_chess_data_generator(validation_data_dir, validation=True, verbose=True)

validation_data_inp = np.empty((0, 8, 8, 6))
validation_data_out = np.empty((0, 32))

for game_input, game_output in validation_data_generator:
    validation_data_inp = np.concatenate((validation_data_inp, game_input))
    validation_data_out = np.concatenate((validation_data_out, game_output))

print "\nValidation data input/output shapes: " + str(validation_data_inp.shape) + " / " + str(validation_data_out.shape)


tensorboard_callback = TensorBoard(log_dir=os.path.join('logs/', name), histogram_freq=1, batch_size=32, write_graph=False, write_grads=True)
checkpoint_callback = ModelCheckpoint(filepath="Models/%s/model_{epoch:02d}_{val_loss:.2f}" % (name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode="min", period=1)

model.fit_generator(generator=ChessParser.cnn_chess_data_generator(path=training_data_dir, validation=False, verbose=False), 
                    steps_per_epoch=100000, epochs=1000000, verbose=1, callbacks=[tensorboard_callback, checkpoint_callback],
                    validation_data=(validation_data_inp, validation_data_out), validation_steps=50, use_multiprocessing=True)