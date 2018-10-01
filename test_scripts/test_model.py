from ChessLogic import ChessLogic
from ChessParser import ChessParser
import pgn
import numpy as np
from random import randint
from keras.models import model_from_json

f = 'Databases/Chess/TrainData/GMallboth.pgn'
str_games = open(f, 'r').read().split('\n\n\n')

# Load model
# model = model_from_json(open('Models/sigmoid-binary_crossentropy-2/model_270000_01-03_10:27:04.json', 'r').read())
# model.load_weights('Models/sigmoid-binary_crossentropy-2/model_270000_01-03_10:27:04.h5')

while True:
    random_game = randint(0, len(str_games) - 2)
    pgn_game = pgn.loads(str_games[random_game])[0]
    print "Game %i: " % random_game + str(pgn_game)

    data = ChessParser.coordinates_for_game(pgn_game)
    coords = np.stack(data[:, 0])
    # Flip black moves
    coords[1::2] = np.flip(coords[1::2], axis=3)
    coords[1::2] = np.flip(coords[1::2], axis=4)
    data[1::2, np.arange(1, 5)] = 7 - data[1::2, np.arange(1, 5)]

    for i, coord in enumerate(coords):
        print (data[i, 1], data[i, 2]), (data[i, 3], data[i, 4])
        print ChessLogic.print_move(coord)
        raw_input()


    random_position = randint(0, len(data) - 1)
    board_position = coords[random_position]
    print "Move %i (%s): " % (random_position, 'White' if random_position%2 == 0 else "Black")
    print ChessLogic.print_move(board_position)

    # Predicted Values
    predicted_values = model.predict(board_position.reshape(1, 896))[0]
    initial_predicted_row = np.argmax(predicted_values[:8])
    initial_predicted_column = np.argmax(predicted_values[8:16])
    final_predicted_row = np.argmax(predicted_values[16:24])
    final_predicted_column = np.argmax(predicted_values[24:])
    print "Predicted Values:"
    print "\t" + str(((initial_predicted_row, initial_predicted_column), (final_predicted_row, final_predicted_column)))

    # Original Values
    print "Original Values:"
    initial_original_row = data[random_position][1]
    initial_original_column = data[random_position][2]
    final_original_row = data[random_position][3]
    final_original_column = data[random_position][4]
    print "\t" + str(((initial_original_row, initial_original_column), (final_original_row, final_original_column)))

    # Comparision
    print "Comparision:"
    print "\tPredicted:\t" + str(((predicted_values[:8][initial_predicted_row],
                                   predicted_values[8:16][initial_predicted_column]),
                                (predicted_values[16:24][final_predicted_row],
                                 predicted_values[24:][final_predicted_column])))

    print "\tOriginal:\t" + str(((predicted_values[:8][initial_original_row],
                                  predicted_values[8:16][initial_original_column]),
                                (predicted_values[16:24][final_original_row],
                                 predicted_values[24:][final_original_column])))

    raw_input()
