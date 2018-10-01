from ChessParser import ChessParser
import numpy as np
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.initializers import glorot_normal
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
import os
import sys

try:
    file_path = sys.argv[1]
except IndexError:
    print "Usage: python cnn_template.py 'config_file'"
    exit()

if not file_path.startswith('/'):
    file_path = os.path.join(os.getcwd(), file_path)

# Prepare gpu
# Disable Tensorflow environments about CPU instructions
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto() 
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config = config))

# Variables
training_data_dir = "Databases/Chess/TrainData"
validation_data_dir = "Databases/Chess/TestData"

# Hyperparameters (read from config-file)
input_shape = (8, 8, 6)

hyperparameters = dict((x, [float(z) for z in y.split(',')]) if ',' in y else (x, int(y)) if y.isdigit() else (x, y) for x, y in [tuple(x.strip().split()) for x in open(file_path, 'r').readlines() if x.strip()])

name = str(hyperparameters['cnn_layers']) + 'cnn_'
for i in range(hyperparameters['cnn_layers']):
    name += str(int(hyperparameters['num_kernels_per_layer'][i])) + '-' + str(int(hyperparameters['kernel_sizes'][i])) + '_'

name += str(hyperparameters['pooling_layers']) + 'pooling_'
for i in range(hyperparameters['pooling_layers']):
    name += str(int(hyperparameters['pooling_layers_sizes'][i])) + '_'

name += str(hyperparameters['hdd_layers']) + 'hdd_'
for i in range(hyperparameters['hdd_layers']):
    name += str(int(hyperparameters['hidden_layers_sizes'][i])) + '_'

name += os.path.splitext(os.path.basename(file_path))[0]

if not os.path.exists('Models/%s' % name):
    os.makedirs('Models/%s' % name)

# First build CNN layers
model = Sequential()
cnn_index = 0
pool_index = 0
for i in range(hyperparameters['cnn_layers'] + hyperparameters['pooling_layers']):

    if i == 0:
        num_kernels = int(hyperparameters['num_kernels_per_layer'][cnn_index])
        kernel_size = int(hyperparameters['kernel_sizes'][cnn_index])
        model.add(Conv2D(num_kernels, (int(kernel_size), int(kernel_size)), padding=hyperparameters['cnn_padding'], activation=hyperparameters['cnn_activation'], input_shape=input_shape, kernel_initializer='glorot_normal'))
        cnn_index += 1
    else:
        if i%(hyperparameters['cnn_layers']/4 + 1) + 1 == hyperparameters['cnn_layers']/4 + 1:
            if pool_index == hyperparameters['pooling_layers'] - 1:
                model.add(MaxPool2D(pool_size=int(hyperparameters['pooling_layers_sizes'][pool_index]), data_format="channels_last"))
            else:
                model.add(MaxPool2D(pool_size=int(hyperparameters['pooling_layers_sizes'][pool_index]), strides=1, data_format="channels_last"))
            pool_index += 1
        else:
            num_kernels = int(hyperparameters['num_kernels_per_layer'][cnn_index])
            kernel_size = int(hyperparameters['kernel_sizes'][cnn_index])
            model.add(Conv2D(num_kernels, (int(kernel_size), int(kernel_size)), padding=hyperparameters['cnn_padding'], activation=hyperparameters['cnn_activation'], kernel_initializer='glorot_normal'))
            cnn_index += 1

# Add hidden layers
model.add(Flatten())
for i in range(hyperparameters['hdd_layers']):
    model.add(Dense(int(hyperparameters['hidden_layers_sizes'][i]), activation=hyperparameters['hdd_activation'], kernel_initializer='glorot_normal'))

def custom_accuracy(y_true, y_pred):
    return K.mean(tf.reduce_all(K.equal(y_true, K.round(y_pred)), axis=1))

model.compile(loss=binary_crossentropy, optimizer='sgd', metrics=[custom_accuracy])

# Save model configurations
def myprint(s):
    with open('Models/%s/summary' % name, 'a+') as f:
        f.write(s)
        f.write('\n')
model.summary(print_fn=myprint)
json = model.to_json()
with open('Models/%s/json' % name, 'w+') as f:
    f.write(json)

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
