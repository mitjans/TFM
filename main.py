from ChessParser import ChessParser
import numpy as np
import tensorflow as tf
from keras_NN import KerasNeuralNetwork
from keras.backend.tensorflow_backend import set_session
import os

# Disable Tensorflow environments about CPU instructions
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto() 
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config = config))

# Variables
training_data_dir = "Databases/Chess/TrainData"
validation_data_dir = "Databases/Chess/TestData"

# Define neural network
neural_network = KerasNeuralNetwork(input_size=896, output_size=32, hidden_layers=20, name='5hd_2do_sigmoid-binary_cross_entropy-2') 

# Prepare validation data
print "Generating validation data..."
validation_data_generator = ChessParser.chess_data_generator(validation_data_dir, validation=True, verbose=True)

validation_data_inp = np.empty((0, 896))
validation_data_out = np.empty((0, 32))

for game_input, game_output in validation_data_generator:
    validation_data_inp = np.concatenate((validation_data_inp, game_input))
    validation_data_out = np.concatenate((validation_data_out, game_output))

print "\nValidation data input/output shapes: " + str(validation_data_inp.shape) + " / " + str(validation_data_out.shape)

# For each database, yield each game
training_data_generator = ChessParser.chess_data_generator(training_data_dir, validation=False, verbose=False)
neural_network.fit_generator(training_generator=training_data_generator, validation_data=(validation_data_inp, validation_data_out))
