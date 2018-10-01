from keras.models import model_from_json
from random import randint
import pgn
from ChessParser import ChessParser
import numpy as np
import json
import sys
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ModelPerformance(object):

    def __init__(self, model_name, json_path, weights_path):
        self.model_name = model_name
        self.model = model_from_json(open(json_path, 'r').read())
        self.model.load_weights(weights_path)
        self.figures_accuracy = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.actual_accuracy = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        # Create ModelPerformance directories
        self.path = '/home/carlesm/TFM/Data/%s' % self.model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def evaluate_model(self, board_pos, player):

        # Get figure at initial position
        predicted_values = self.model.predict(board_pos)[0]

        initial_predicted_row = np.argmax(predicted_values[:8])
        initial_predicted_column = np.argmax(predicted_values[8:16])
        final_predicted_row = np.argmax(predicted_values[16:24])
        final_predicted_column = np.argmax(predicted_values[24:])

        initial_pos = (initial_predicted_row, initial_predicted_column)
        figure_moving = np.where(board_pos[initial_pos] == (-1 if player else 1))[0]
        
        if len(figure_moving) > 1: raise TypeError

        if not figure_moving.size:
            figure_moving = 0
        else: figure_moving = figure_moving[0]

        # Save relative vector
        self.figures_accuracy[figure_moving].append((initial_predicted_row - final_predicted_row,
                                                    final_predicted_column - initial_predicted_column))

    def save_evaluation(self, suffix):
        with open(os.path.join(self.path, 'performance_%s.json' % (suffix)), 'w') as model_evaluation_file:
            json.dump(self.figures_accuracy, model_evaluation_file)
            model_evaluation_file.write('\n')

    def save_actual_movements(self, suffix):
        with open(os.path.join(self.path, 'correct_%s.json' % (suffix)), 'w') as model_evaluation_file:
            json.dump(self.actual_accuracy, model_evaluation_file)
            model_evaluation_file.write('\n')


    def correct_movements(self, board_pos, initial_pos, final_pos, player):
        player_board = board_pos[player,:]
        player_board = np.transpose(player_board, (1, 2, 0))
        figure_moving = np.where(player_board[initial_pos] == 1)[0][0]

        self.actual_accuracy[figure_moving].append((initial_pos[0] - final_pos[0],
                                                     final_pos[1] - initial_pos[1]))


games_path = '/home/carlesm/TFM/Databases/Chess/TestData/validation_data.pgn'
models_path = '/home/carlesm/TFM/Models/evaluated'
games = open(games_path).read().split('\n\n\n')

for model_dir in os.listdir(models_path):
    model = '_'.join(model_dir.split('_')[-4:])
    model_frames = sorted([x for x in os.listdir(os.path.join(models_path, model_dir)) if 'model' in x])
    print model_frames
    print "[PYTHON] Working with model '" + model + "'..."

    for frame_index, model_frame in enumerate(model_frames):
        model_json = os.path.join(models_path, model_dir, "json")
        model_performance = ModelPerformance(model, model_json, os.path.join(models_path, model_dir, model_frame))

        for i in range(10000):
            sys.stdout.write("\r[PYTHON]\tGenerating '" + model_frame + "' data [%d/%d]..." % (i+1, 10000))
            sys.stdout.flush()

            random_game_number = randint(0, len(games) - 2)
            try:
                random_game = pgn.loads(games[random_game_number])[0]
            except AttributeError:
                continue

            data = ChessParser.coordinates_for_game(random_game)

            if not data.size:
                continue            
            # Flip black moves
            moves = np.stack(data[:, 1])
            moves[1::2] = np.flip(moves[1::2], axis=3)
            moves[1::2] = np.flip(moves[1::2], axis=4)

            random_move_number = randint(0, len(moves) - 1)

            board_pos = moves[random_move_number]
            board_pos = np.expand_dims(np.transpose((board_pos.board[1] - board_pos.board[0])[np.arange(1,7)], axes = (1, 2, 0)), axis=0)

            initial_pos = (data[random_move_number][2], data[random_move_number][3])
            final_pos = (data[random_move_number][4], data[random_move_number][5])

            try:
                model_performance.evaluate_model(board_pos, int(random_move_number % 2 == 0))
            except TypeError:
                print
                print random_game_number
                print random_game
                print random_move_number
                print "DONE"
                exit()


        model_performance.save_evaluation(str("%010d" % frame_index))
print


print "[PYTHON] DONE"
