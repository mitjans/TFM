from ChessLogic import ChessLogic
from keras.models import model_from_json
import numpy as np
from pprint import pprint
from copy import deepcopy

def outer_mean(vec1, vec2):
    return np.add.outer(vec1, vec2)/2

# Load model
model_json = "Models/json"
model_weights = "Models/model_101_0.24"
model = model_from_json(open(model_json, 'r').read())
model.load_weights(model_weights)


game = ChessLogic()
player = 1

while True:
    print game.board.shape
    input_data = np.expand_dims(np.transpose((game.board[1] - game.board[0])[np.arange(1,7)], axes = (1, 2, 0)), axis=0)
    predicted_values = model.predict(input_data)[0]
    player_board = game.board[player,:]

    mask = player_board[0]
    initial_matrix = outer_mean(predicted_values[:8], predicted_values[8:16])
    # pprint(np.flipud(np.fliplr(np.round(initial_matrix, 2))))

    initial_matrix_masked = np.ma.masked_array(initial_matrix, mask).filled(0)
    
    # pprint(np.flipud(np.fliplr(np.round(initial_matrix_masked, 2))))

    game_copy = deepcopy(game)
    while True:

        initial_coords = np.unravel_index(initial_matrix_masked.argmax(), initial_matrix_masked.shape)
        figure = ChessLogic.figure_for_player_at_position(game.board, 1, initial_coords)

        # Masking final matrix
        mask = np.ones((8, 8))
        for i in range(8):
            for j in range(8):
                if game_copy.is_valid_movement(figure, (initial_coords[0], initial_coords[1]), (i, j)):
                    mask[i, j] = 0
        
        if not np.count_nonzero(mask):
            initial_matrix_masked[initial_coords] = 0
        else:
            break

    final_matrix = outer_mean(predicted_values[16:24], predicted_values[24:])
    # pprint(np.flipud(np.fliplr(np.round(final_matrix, 2))))

    final_matrix_masked = np.ma.masked_array(final_matrix, mask).filled(0)
    final_coords = np.unravel_index(final_matrix_masked.argmax(), final_matrix_masked.shape)
    
    # pprint(np.flipud(np.fliplr(np.round(final_matrix_masked, 2))))

    if not game.move(initial_coords, final_coords, 5):
        print initial_coords, final_coords
        exit()
    
    flipped_board = np.flip(np.flip(game.board, axis=2), axis=3)

    print ChessLogic.print_move(flipped_board)

    initial_row, initial_col, final_row, final_col = [int(x) for x in raw_input("Enter move: ").split()]

    if not game.move((initial_row, initial_col), (final_row, final_col),  5):
        print (initial_row, initial_col), (final_row, final_col)
        exit()
