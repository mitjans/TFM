import pgn
from ChessLogic import ChessLogic
import re
import numpy as np
import sys
import os
from random import randint, choice
from copy import deepcopy

pgn_figure_dict = {'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6, 'O': 6}
column_to_index_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
row_to_index_dict = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}

debug = True


class ChessParser(object):

    @staticmethod
    def single_pgn_to_coord(pgn_str):
        """
        Given a location in PGN notation, convert it to valid coordinates
        :param pgn_str: (string) PGN location in board (i.e b3)
        :return: (1x2 tuple) Coordinates (i.e (5, 0)
        """
        column = column_to_index_dict[pgn_str[0]]
        row = row_to_index_dict[int(pgn_str[1])]
        return row, column

    @staticmethod
    def coordinates_for_game(pgn_game):
        """
        Given a PGNGame object, returns the game board previous to the move,
        the move itself and the promotion in the following form:
        [game.board, (ori_y, ori_x), (fin_y, fin_x), promotion]
            - game.board: This will be the input to the Neural Network
            - (ori_y, ori_x), (fin_y, fin_x) and promotion will be the output

        :param pgn_game: (PGNGame) Game from which movements will be returned
        :return: (list of tuples) [game.board, initial_pos, final_pos, promotion
        """

        return_list = []

        # Start Game Logic
        game = ChessLogic()

        # Get moves from PGNGame object, removing final move (game result)
        valid_moves = pgn_game.moves[:-1]

        for move in valid_moves:
            # We are only interested in alfanumeric characters. It also removes 'x' from the move (if present).
            # 'x' denotes a kill, which the game logic will take care of it.
            pattern = re.compile(r'[\W_x]+')
            move = pattern.sub('', move)

            if not move:
                continue

            # It seems there are some databases that indicate 'en passant' kills with 'ep' at the end. We remove it
            if move.endswith('ep'):
                move = move[:-2]

            # Define variables
            initial_column = 0
            initial_row = 0
            final_column = 0
            final_row = 0
            promotion = 0

            # Check if moving Pawn
            if not move[0].isupper():
                if len(move) == 2:
                    final_row, final_column = ChessParser.single_pgn_to_coord(move)
                    initial_column = final_column

                    possible_rows = game.rows_for_player_figure_at_column(game.current_player, 1, initial_column)

                    # When there's ambiguity, PGN will only inform if the two figures are in the same row
                    # This means that when two figures are in the same column, we need to move the correct one
                    # (The only one that can make the move)
                    for possible_row in possible_rows:
                        straight_range = ChessLogic.straight_coordinate_range((possible_row, initial_column),
                                                                              (final_row, final_column))
                        should_continue = False
                        for new_pos in straight_range:
                            if not game.is_position_empty(new_pos) or (
                                    new_pos[0] < possible_row and game.current_player == 0) or (
                                    new_pos[0] > possible_row and game.current_player == 1):
                                should_continue = True
                                break

                        if should_continue:
                            continue
                        else:
                            initial_row = possible_row
                            break

                elif len(move) == 4:
                    # A pawn is killing and it is promoting at the same time (i.e ba1Q)
                    promotion = pgn_figure_dict[move[3]]
                    move = move[:3]

                if len(move) == 3:

                    # When a pawn is promoted, the resulting string will be like f1Q,
                    # where f1 indicates the destination and Q the promotion
                    if move[2].isupper():
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[:2])
                        initial_column = final_column
                        possible_rows = game.rows_for_player_figure_at_column(game.current_player, 1, final_column)

                        # When there's ambiguity, PGN will only inform if the two figures are in the same row
                        # This means that when two figures are in the same column, we need to move the correct one
                        # (The only one that can make the move)
                        for possible_row in possible_rows:
                            straight_range = ChessLogic.straight_coordinate_range((possible_row, initial_column),
                                                                                  (final_row, final_column))
                            should_continue = False
                            for new_pos in straight_range:
                                if not game.is_position_empty(new_pos):
                                    should_continue = True
                                    break
                            if should_continue:
                                continue
                            else:
                                initial_row = possible_row
                                break

                        promotion = pgn_figure_dict[move[2]]
                    else:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                        initial_column = column_to_index_dict[move[0]]

                        possible_rows = game.rows_for_player_figure_at_column(game.current_player, 1, initial_column)
                        diagonal_locs = ChessLogic.valid_diagonal_pawn_positions_for_pawn_at_location(
                            (final_row, final_column))

                        for diagonal_loc in diagonal_locs:
                            found = False
                            for possible_row in possible_rows:
                                if diagonal_loc[0] == possible_row:
                                    if ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                          (possible_row, initial_column)) == 1:
                                        if game.current_player == 1 and possible_row < final_row or game.current_player == 0 and possible_row > final_row:
                                            continue
                                        else:
                                            initial_row = possible_row
                                            found = True
                                            break
                            if found:
                                break

            else:
                # Moving a non-Pawn figure
                figure = move[0]

                if figure == 'R':
                    if len(move) == 3:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                        for valid_rook_move in game.valid_rook_positions_for_rook_at_location(
                                (final_row, final_column)):
                            location_figure = ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                                 valid_rook_move)
                            if location_figure == 2 and not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                    valid_rook_move,
                                    (final_row, final_column)):
                                initial_row, initial_column = valid_rook_move
                    elif len(move) == 4:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[2:])
                        if move[1].isalpha():
                            initial_column = column_to_index_dict[move[1]]
                            possible_rows = game.rows_for_player_figure_at_column(game.current_player, 2,
                                                                                  initial_column)

                            for possible_row in possible_rows:
                                if ChessLogic.are_coordinates_straight((possible_row, initial_column),
                                                                       (final_row, final_column)):
                                    if not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                            (possible_row, initial_column), (final_row, final_column)):
                                        initial_row = possible_row
                                        break
                        else:
                            initial_row = row_to_index_dict[int(move[1])]
                            possible_columns = game.columns_for_player_figure_at_row(game.current_player, 2,
                                                                                     initial_row)
                            for possible_column in possible_columns:
                                if ChessLogic.are_coordinates_straight((initial_row, possible_column),
                                                                       (final_row, final_column)):
                                    if not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                            (initial_row, possible_column), (final_row, final_column)):
                                        initial_column = possible_column
                                        break
                    elif len(move) == 5:
                        initial_row, initial_column = ChessParser.single_pgn_to_coord(move[1:3])
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[3:])

                elif figure == 'N':
                    # If not specifying more information (i.e Nb3)
                    if len(move) == 3:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                        for valid_knight_move in game.valid_knight_positions_for_knight_at_location(
                                (final_row, final_column)):
                            location_figure = ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                                 valid_knight_move)
                            if location_figure == 3 and not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                    valid_knight_move, (final_row, final_column)):
                                initial_row, initial_column = valid_knight_move
                                break

                    # Specifying Knight position
                    elif len(move) == 4:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[2:])
                        if move[1].isalpha():
                            initial_column = column_to_index_dict[move[1]]
                            possible_rows = game.rows_for_player_figure_at_column(game.current_player, 3,
                                                                                  initial_column)
                            for possible_row in possible_rows:
                                if not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        (possible_row, initial_column), (final_row, final_column)):
                                    initial_row = possible_row

                        else:
                            initial_row = row_to_index_dict[int(move[1])]
                            possible_columns = game.columns_for_player_figure_at_row(game.current_player, 3,
                                                                                     initial_row)

                            for possible_column in possible_columns:
                                if not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        (initial_row, possible_column), (final_row, final_column)):
                                    initial_column = possible_column

                    elif len(move) == 5:
                        initial_row, initial_column = ChessParser.single_pgn_to_coord(move[1:3])
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[3:])

                elif figure == 'B':
                    # If not specifying more information (i.e Bb3)
                    if len(move) == 3:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                        for valid_bishop_move in game.valid_bishop_positions_for_bishop_at_location(
                                (final_row, final_column)):
                            location_figure = ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                                 valid_bishop_move)
                            if location_figure == 4 and not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                    valid_bishop_move, (final_row, final_column)):
                                initial_row, initial_column = valid_bishop_move

                    # Specifying Bishop position
                    elif len(move) == 4:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[2:])
                        if move[1].isalpha():
                            initial_column = column_to_index_dict[move[1]]
                            possible_rows = game.rows_for_player_figure_at_column(game.current_player, 4,
                                                                                  initial_column)

                            for possible_row in possible_rows:
                                possible_initial_position = np.array((possible_row, initial_column))
                                diagonal_range = ChessLogic.diagonal_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))

                                diagonal_range_empty = True
                                for diagonal_loc in diagonal_range[:-1]:
                                    if not game.is_position_empty(diagonal_loc):
                                        diagonal_range_empty = False

                                if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        possible_initial_position, (final_row, final_column)):
                                    diagonal_range_empty = False

                                if diagonal_range_empty:
                                    initial_row = possible_row
                                    break

                        else:
                            initial_row = row_to_index_dict[int(move[1])]
                            possible_columns = game.columns_for_player_figure_at_row(game.current_player, 4,
                                                                                     initial_row)
                            for possible_column in possible_columns:
                                possible_initial_position = np.array((initial_row, possible_column))
                                diagonal_range = ChessLogic.diagonal_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))

                                diagonal_range_empty = True
                                for diagonal_loc in diagonal_range[:-1]:
                                    if not game.is_position_empty(diagonal_loc):
                                        diagonal_range_empty = False

                                if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        possible_initial_position, (final_row, final_column)):
                                    diagonal_range_empty = False

                                if diagonal_range_empty:
                                    initial_column = possible_column
                                    break

                    elif len(move) == 5:
                        initial_row, initial_column = ChessParser.single_pgn_to_coord(move[1:3])
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[3:])

                elif figure == 'Q':
                    if len(move) == 3:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                        for valid_queen_move in game.valid_queen_positions_for_queen_at_location(
                                (final_row, final_column)):
                            location_figure = ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                                 valid_queen_move)
                            if location_figure == 5 and not game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                    valid_queen_move, (final_row, final_column)):
                                initial_row, initial_column = valid_queen_move
                    # If a pawn is promoted to queen, the possibility of ambiguity between two queens is present
                    elif len(move) == 4:
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[2:])
                        if move[1].isalpha():
                            initial_column = column_to_index_dict[move[1]]
                            possible_rows = game.rows_for_player_figure_at_column(game.current_player, 5,
                                                                                  initial_column)

                            for possible_row in possible_rows:
                                possible_initial_position = np.array((possible_row, initial_column))
                                straight_range = ChessLogic.straight_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))
                                diagonal_range = ChessLogic.diagonal_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))

                                straight_range_empty = True
                                for straight_loc in straight_range[:-1]:
                                    if not game.is_position_empty(straight_loc):
                                        straight_range_empty = False

                                if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        possible_initial_position, (final_row, final_column)):
                                    straight_range_empty = False

                                if straight_range_empty:
                                    initial_row = possible_row
                                    break

                                else:
                                    diagonal_range_empty = True
                                    for diagonal_loc in diagonal_range[:-1]:
                                        if not game.is_position_empty(diagonal_loc):
                                            diagonal_range_empty = False

                                    if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                            possible_initial_position, (final_row, final_column)):
                                        diagonal_range_empty = False

                                    if diagonal_range_empty:
                                        initial_row = possible_row
                                        break

                        else:
                            initial_row = row_to_index_dict[int(move[1])]
                            possible_columns = game.columns_for_player_figure_at_row(game.current_player, 5,
                                                                                     initial_row)
                            for possible_column in possible_columns:
                                possible_initial_position = np.array((initial_row, possible_column))
                                straight_range = ChessLogic.straight_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))
                                diagonal_range = ChessLogic.diagonal_coordinate_range(possible_initial_position,
                                                                                      (final_row, final_column))

                                straight_range_empty = True
                                for straight_loc in straight_range[:-1]:
                                    if not game.is_position_empty(straight_loc):
                                        straight_range_empty = False

                                if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                        possible_initial_position, (final_row, final_column)):
                                    straight_range_empty = False

                                if straight_range_empty:
                                    initial_column = possible_column
                                    break

                                else:
                                    diagonal_range_empty = True
                                    for diagonal_loc in diagonal_range[:-1]:
                                        if not game.is_position_empty(diagonal_loc):
                                            diagonal_range_empty = False

                                    if game.will_player_be_in_check_after_removing_figure_at_location_to_location(
                                            possible_initial_position, (final_row, final_column)):
                                        diagonal_range_empty = False

                                    if diagonal_range_empty:
                                        initial_column = possible_column
                                        break

                    elif len(move) == 5:
                        initial_row, initial_column = ChessParser.single_pgn_to_coord(move[1:3])
                        final_row, final_column = ChessParser.single_pgn_to_coord(move[3:])

                elif figure == 'K':
                    final_row, final_column = ChessParser.single_pgn_to_coord(move[1:])
                    for valid_king_move in game.valid_king_positions_for_king_at_location((final_row, final_column)):
                        location_figure = ChessLogic.figure_for_player_at_position(game.board, game.current_player,
                                                                             valid_king_move)
                        if location_figure == 6:
                            initial_row, initial_column = valid_king_move

                elif figure == 'O':
                    if len(move) == 2:
                        final_column = 6
                        final_row = 7 * game.current_player
                        initial_column = 4
                        initial_row = 7 * game.current_player
                    else:
                        final_column = 2
                        final_row = 7 * game.current_player
                        initial_column = 4
                        initial_row = 7 * game.current_player

                # There are some unknown final movements (i.e Z0). Ignore them
                else:
                    break

            initial_pos = (initial_row, initial_column)
            final_pos = (final_row, final_column)

            # Before performing the move in the game board, we keep a copy of it for yielding
            board_copy = np.copy(game.board)
            game_copy = deepcopy(game)

            # Apply movement, if it detects any error, it will stop parsing this game
            if not game.move(initial_pos, final_pos, promotion):
                break

            return_list.append([game_copy, board_copy, initial_pos[0], initial_pos[1], final_pos[0], final_pos[1]])

        return np.asarray(return_list)

    @staticmethod
    def games_in_file(file_path):
        """
        Given apgn file path, it returns a list with all games converted to PGNGame
        :param file_path: (string) pgn file path
        :return: (list) List of PGNGames
        """

        try:
            pgn_file = open(file_path, 'r')
        except IOError:
            return

        str_games = pgn_file.read().split('\n\n\n')

        for str_game in str_games:
            # There are empty games with 3 white lines between events and results, which 'split' splits.
            # Remove empty games
            try:
                pgn_game = pgn.loads(str_game)
            except AttributeError:
                continue

            if len(str_game) == 0:
                continue

            yield pgn_game[0]

    @staticmethod
    def chess_data_generator(path, validation, verbose):
        delim = "[Event "
        files = os.listdir(path)

        while 1:
            for i, database_filename in enumerate(files):
                pgn_file = open(os.path.join(path, database_filename), 'r')
                str_games = pgn_file.read().split(delim)

                for j, str_game in enumerate(str_games):
                    if verbose:
                        sys.stdout.write("Database %03d/%03d - Game %06d/%06d\r" % (i + 1, len(files), j + 1, len(str_games)))
                        sys.stdout.flush()
                    if len(str_game) == 0:
                        continue

                    try:
                        pgn_game = pgn.loads(delim + str_game)
                    except AttributeError:
                        continue

                    data = ChessParser.coordinates_for_game(pgn_game[0])

                    if not data.size:
                        continue

                    coords = np.stack(data[:, 1])

                    # Flip black moves
                    coords[1::2] = np.flip(coords[1::2], axis=3)
                    coords[1::2] = np.flip(coords[1::2], axis=4)
                    data[1::2, np.arange(2, 6)] = 7 - data[1::2, np.arange(2, 6)]

                    input_data = coords.reshape(coords.shape[0], 896)

                    temp = (np.arange(8) == data[:, [2, 3, 4, 5]][..., None]).astype(int)
                    output_data = temp.reshape(data.shape[0], 32)


                    yield (input_data, output_data)

            if validation:
                break

    @staticmethod
    def random_chess_game_from_path(path):
        delim = "[Event "
        files = os.listdir(path)
        random_file = randint(0, len(os.listdir(path)) - 1)
        pgn_file = open(os.path.join(path, files[random_file]), 'r')
        str_games = pgn_file.read().split(delim)
        return delim + choice(str_games)

    @staticmethod
    def cnn_chess_data_generator(path, validation, verbose):
        generator = ChessParser.chess_data_generator(path, validation, verbose)
        for (input_data, output_data) in generator: 
            input_data = input_data.reshape(input_data.shape[0], 2, 7, 8, 8)
            input_data = np.transpose((input_data[:,1] - input_data[:,0])[:,np.arange(1,7)], axes=(0, 2, 3, 1))
            yield (input_data, output_data)
