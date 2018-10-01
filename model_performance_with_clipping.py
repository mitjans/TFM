from keras.models import model_from_json
from random import randint
import pgn
from ChessParser import ChessParser
from ChessLogic import ChessLogic
import numpy as np
import json
import sys
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def outer_mean(vec1, vec2):
    return np.add.outer(vec1, vec2)/2

class ModelPerformance(object):

    def __init__(self, model_path, weights_path):
        self.model_name = os.path.split(model_path)[1]
        self.model = model_from_json(open(model_path, 'r').read())
        self.model.load_weights(weights_path)
        self.figures_accuracy = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.actual_accuracy = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        # Create ModelPerformance directories
        self.path = 'Data/%s' % self.model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def evaluate_model(self, game, board_pos, player):

        # Get figure at initial position
        player_board = board_pos[player,:]

        predicted_values = self.model.predict(board_pos.reshape(1, 896))[0]

        # Masking initial matrix
        mask = player_board[0]
        initial_matrix = outer_mean(predicted_values[:8], predicted_values[8:16])            
        initial_matrix_masked = np.ma.masked_array(initial_matrix, mask).filled(0)

        while True:
            initial_coords = np.unravel_index(initial_matrix_masked.argmax(), initial_matrix_masked.shape)

            figure = ChessLogic.figure_for_player_at_position(board_pos, player, initial_coords)
                    
            # Masking final matrix
            mask = np.ones((8, 8))
            for i in range(8):
                for j in range(8):
                    if player == 0:
                        valid = game.is_valid_movement(figure, (7 - initial_coords[0], 7 - initial_coords[1]), (i, j))
                    else:
                        valid = game.is_valid_movement(figure, (initial_coords[0], initial_coords[1]), (i, j))
                    if valid:
                        if player == 0:
                            mask[7 - i, 7 - j] = 0
                        else:
                            mask[i, j] = 0
            
            if not np.count_nonzero(mask):
                initial_matrix_masked[initial_coords] = 0
            else:
                break

        final_matrix = outer_mean(predicted_values[16:24], predicted_values[24:])
        final_matrix_masked = np.ma.masked_array(final_matrix, mask).filled(0)
        final_coords = np.unravel_index(final_matrix_masked.argmax(), final_matrix_masked.shape)

        print "Predicted: ", initial_coords, final_coords
        # Save relative vector
        self.figures_accuracy[figure].append((initial_coords[1] - final_coords[1],
                                                     initial_coords[0] - final_coords[0]))

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


games_path = '/Users/carlesm/Downloads/KingBase2018-pgn-2'
models_path = 'Models'

for model in os.listdir(models_path):
    if os.path.isfile(os.path.join(models_path, model)):
        continue
    model_frames = os.listdir(os.path.join(models_path, model, "Frames"))
    print "[PYTHON] Working with model '" + model + "'..."
    for frame_index, model_frame in enumerate(model_frames):
        sys.stdout.write("\r[PYTHON]\tGenerating '" + model_frame + "' data...\n")
        sys.stdout.flush()

        # Check if performance has been already computed
        model_json = os.path.join(models_path, os.path.join(model, "cnn_model.json"))
        model_weights = os.path.join(models_path, os.path.join(model, "Frames/" + model_frame))

        model_performance = ModelPerformance(model_json, model_weights)

        for i in range(1000):

            try:
                random_game = pgn.loads(ChessParser.random_chess_game_from_path(games_path))[0]
            except AttributeError:
                continue

            data = ChessParser.coordinates_for_game(random_game)

            if not data.size:
                continue

            moves = np.stack(data[:, 1])
            # Flip black moves
            data[1::2, np.arange(2, 6)] = 7 - data[1::2, np.arange(2, 6)]
            moves[1::2] = np.flip(moves[1::2], axis=3)
            moves[1::2] = np.flip(moves[1::2], axis=4)

            random_move_number = randint(0, len(moves) - 1)

            game_logic = data[random_move_number, 0]
            board_pos = moves[random_move_number]
            
            print ChessLogic.print_move(board_pos)

            initial_pos = (data[random_move_number][2], data[random_move_number][3])
            final_pos = (data[random_move_number][4], data[random_move_number][5])
            print "Actual: ", initial_pos, final_pos

            model_performance.evaluate_model(game_logic, board_pos, int(random_move_number % 2 == 0))
            
            print
            raw_input()

        model_performance.save_evaluation(str("%010d" % frame_index))
print
