import numpy as np


# noinspection SpellCheckingInspection
class ChessLogic(object):
    """
    Representation of a Chess Game in progress.
    Instance Variables:
            board:
                numpy array of size 2x7x8x8 (7 - one-hot vector of figures)
                None    [1, 0, 0, 0, 0, 0, 0]
                Pawn    [0, 1, 0, 0, 0, 0, 0]
                Rook    [0, 0, 1, 0, 0, 0, 0]
                Knight  [0, 0, 0, 1, 0, 0, 0]
                Bishop  [0, 0, 0, 0, 1, 0, 0]
                Queen   [0, 0, 0, 0, 0, 1, 0]
                King    [0, 0, 0, 0, 0, 0, 1]
            white_en_passant:
                coordinates where black can kill white en passant
            black_en_passant:
                coordinates where white can kill black en passant
            black_can_castle:
                wether black can castle or not
            white_can_caste
                wether white can castle or not
            castle:
                Did the last player castle?
                    0 - No
                    1 - Small
                    2 - Large
            kill:
                Wether there was a kill or not in the last play
            current_player:
                False for Black, True for White
            promotion:
                Did the last player promote a pawn?
    """

    def __init__(self):
        self.board = np.zeros((2, 7, 8, 8))
        self.white_en_passant = None
        self.black_en_passant = None
        self.black_can_small_castle = True
        self.white_can_small_castle = True
        self.black_can_large_castle = True
        self.white_can_large_castle = True
        self.castle = 0
        self.kill = False
        self.current_player = True
        self.promotion = False

        self.prepare()

    def prepare(self):
        """
        Initialize the board position
        :return: None
        """

        white_board = self.board[1]
        black_board = self.board[0]

        # None Figures
        white_board[0][:6] = 1
        black_board[0][2:8] = 1
        # Pawns
        white_board[1][6] = 1
        black_board[1][1] = 1
        # Rooks
        white_board[2][7][0] = white_board[2][7][7] = 1
        black_board[2][0][0] = black_board[2][0][7] = 1
        # Knights
        white_board[3][7][1] = white_board[3][7][6] = 1
        black_board[3][0][1] = black_board[3][0][6] = 1
        # Bishops
        white_board[4][7][2] = white_board[4][7][5] = 1
        black_board[4][0][2] = black_board[4][0][5] = 1
        # Queens
        white_board[5][7][3] = 1
        black_board[5][0][3] = 1
        # Kings
        white_board[6][7][4] = 1
        black_board[6][0][4] = 1

    def move(self, initial_position, final_position, promotion_figure):
        """
        Move a player's figure to a desired position

        :param initial_position: (1x3 tuple) Figure's current position
        :param final_position: (1x3 tuple) Figure's final destination
        :param promotion_figure: (int) In case of promotion, figure to promote. Default queen
        :return: (bool) True if the movement was applied, False otherwise
        """

        # Get figure trying to move
        figure = ChessLogic.figure_for_player_at_position(self.board, self.current_player, initial_position)

        # If trying to move a None-Figure, False
        if not figure:
            return False

        # If movement is not valid, False
        if not self.is_valid_movement(figure, initial_position, final_position):
            return False

        # Reset en_passant varialble
        if self.white_en_passant and self.current_player == 0:
            self.white_en_passant = None
        elif self.black_en_passant and self.current_player == 1:
            self.black_en_passant = None

        # Update board
        self.board[int(self.current_player)][0][initial_position] = 1
        self.board[int(self.current_player)][0][final_position] = 0
        self.board[int(self.current_player)][figure][initial_position] = 0
        self.board[int(self.current_player)][figure][final_position] = 1

        # If the movement implied a promotion, change board position accordingly
        if self.promotion:
            self.board[int(self.current_player)][promotion_figure][final_position] = 1
            self.board[int(self.current_player)][figure][final_position] = 0
            self.promotion = False
        # Else if movement implied a kill
        if self.kill:
            killed_figure = ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_position)
            self.board[int(not self.current_player)][0][final_position] = 1
            self.board[int(not self.current_player)][killed_figure][final_position] = 0
            self.kill = False
        # Else if small castle
        elif self.castle == 1:
            x = 7 if self.current_player else 0

            self.board[int(self.current_player)][0][(x, 7)] = 1
            self.board[int(self.current_player)][0][(x, 5)] = 0
            self.board[int(self.current_player)][2][(x, 7)] = 0
            self.board[int(self.current_player)][2][(x, 5)] = 1

            self.castle = 0
        # Else if large castle
        elif self.castle == 2:
            x = 7 if self.current_player else 0

            self.board[int(self.current_player)][0][(x, 0)] = 1
            self.board[int(self.current_player)][0][(x, 3)] = 0
            self.board[int(self.current_player)][2][(x, 0)] = 0
            self.board[int(self.current_player)][2][(x, 3)] = 1

            self.castle = 0

        # Change current Player
        self.current_player = not self.current_player

        return True

    def is_valid_movement(self, figure, initial_pos, final_pos):
        """
        Given a figure and an initial position, returns if it's possible for the figure to move to the final position

        :param figure: (int) Figure to move {P: 1, R: 2, N: 3, B: 4, Q: 5, K: 6}
        :param initial_pos: (1x3 tuple) initial position
        :param final_pos: (1x3 tuple) final position
        :return: (bool) True if movement is valid, False otherwise
        """

        # If trying to move a non-figure position, False
        if not figure:
            return False
        # If trying to move outbounds, False
        if (np.array(final_pos) < 0).any() or (np.array(final_pos) > 7).any():
            return False

        # Get straight and diagonal ranges for movement
        straight_range = self.straight_coordinate_range(initial_pos, final_pos)
        diagonal_range = self.diagonal_coordinate_range(initial_pos, final_pos)

        # Pawn
        if figure == 1:
            # If trying to move forward
            if straight_range.size and ((final_pos[0] > initial_pos[0] and self.current_player == 0) or
                                        (final_pos[0] < initial_pos[0] and self.current_player == 1)):
                # If trying to move one step forward
                if straight_range.shape[0] == 1:
                    # Check if position is empty
                    if self.is_position_empty(final_pos):
                        # If reaching last rank, promotion
                        if final_pos[0] == 7*(not self.current_player):
                            self.promotion = True
                        return True
                    return False
                # Else if trying to move two steps forward
                elif straight_range.shape[0] == 2:
                    # If not in second row, False
                    if (initial_pos[0] != 1 and self.current_player == 0) or (self.current_player == 1 and initial_pos[0] != 6):
                        return False
                    for coord in straight_range:
                        if not self.is_position_empty(coord):
                            return False
                    # If movement is correct, there is a possibility that an en passant kill takes place
                    if self.current_player == 1:
                        self.white_en_passant = list(straight_range[0])
                    else:
                        self.black_en_passant = list(straight_range[0])
                    return True
                # Else (trying to move more than 2 steps forward)
                return False
            # Else trying to kill
            elif diagonal_range.shape[0] == 1 and ((final_pos[0] > initial_pos[0] and self.current_player == 0) or
                                                   (final_pos[0] < initial_pos[0] and self.current_player == 1)):
                # If currently playing white and black_en_passant is final_pos, kill
                if self.current_player == 1 and np.array_equal(self.black_en_passant, final_pos):
                    # In order to kill properly, we need to move back the enemy pawn to final_pos
                    # We need to change board position
                    self.board[int(not self.current_player)][0][final_pos] = 0
                    self.board[int(not self.current_player)][0][final_pos[0] + 1][final_pos[1]] = 1
                    self.board[int(not self.current_player)][1][final_pos] = 1
                    self.board[int(not self.current_player)][1][final_pos[0] + 1][final_pos[1]] = 0

                    self.kill = True
                    return True
                # Else if currently playing black and white_en_passant is final_pos, kill
                elif self.current_player == 0 and np.array_equal(self.white_en_passant, final_pos):
                    # In order to kill properly, we need to move back the enemy pawn to final_pos
                    # We need to change board position
                    self.board[int(not self.current_player)][0][final_pos] = 0
                    self.board[int(not self.current_player)][0][final_pos[0] - 1][final_pos[1]] = 1
                    self.board[int(not self.current_player)][1][final_pos] = 1
                    self.board[int(not self.current_player)][1][final_pos[0] - 1][final_pos[1]] = 0

                    self.kill = True
                    return True
                # Else if enemy figure in final_pos, kill
                elif ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                    # If killing in last line, promotion!
                    if self.current_player == 1 and final_pos[0] == 0 or self.current_player == 0 and final_pos[0] == 7:
                        self.promotion = True
                    self.kill = True
                    return True
                # Pawn not capable of killing
                return False

            # Else (not a valid move for pawn)
            return False

        # Rook
        elif figure == 2:
            # If trying to move to same position, False
            if not straight_range.size:
                return False
            # Else if no figures in the way
            for coord in straight_range[:-1]:
                if not self.is_position_empty(coord):
                    return False
            # If there is an ally figure at final_pos, False
            if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                return False
            # If code execution reaches this point, the movement is valid, so player cannot castle
            if self.current_player == 1:
                # If moving left rook, player cannot large castle
                if np.array_equal(initial_pos, (7, 0)):
                    self.white_can_large_castle = False
                # Else if moving right rook, player cannot small castle
                elif np.array_equal(initial_pos, (7, 7)):
                    self.white_can_small_castle = False

            elif self.current_player == 0:
                # If moving left rook, player cannot large castle
                if np.array_equal(initial_pos, (0, 0)):
                    self.black_can_large_castle = False
                # Else if moving right rook, player cannot small castle
                elif np.array_equal(initial_pos, (0, 7)):
                    self.black_can_small_castle = False

            # If there is an enemy figure at final_pos, kill
            if ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                self.kill = True
            return True

        # Knight
        elif figure == 3:
            # Check if movement is valid for knight
            if (abs(final_pos[0] - initial_pos[0]) == 2 and abs(final_pos[1] - initial_pos[1]) == 1) or\
                    (abs(final_pos[0] - initial_pos[0]) == 1 and abs(final_pos[1] - initial_pos[1]) == 2):
                # If there is an ally figure at final_pos, False
                if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                    return False
                # Else if there is an enemy figure at final_pos, kill
                if ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                    self.kill = True
                return True
            # Else (not valid move for Knight)
            return False

        # Bishop
        elif figure == 4:
            # If trying to move to same position, False
            if not diagonal_range.size:
                return False
            # Else if no figures in the way
            for coord in diagonal_range[:-1]:
                if not self.is_position_empty(coord):
                    return False
            # If there is an ally figure at final_pos, False
            if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                return False
            # If code execution reaches this point, the movement is valid, ergo True
            # If there is an enemy figure at final_pos, kill
            elif ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                self.kill = True
            return True

        # Queen
        elif figure == 5:
            # If trying to move in a straight line
            if straight_range.size:
                for coord in straight_range[:-1]:
                    if not self.is_position_empty(coord):
                        return False
                # If there is an ally figure at final_pos, False
                if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                    return False
                # If code execution reaches this point, the movement is valid, ergo True
                # If there is an enemy figure at final_pos, kill
                elif ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                    self.kill = True
                return True

            # Else if trying to move in a diagonal line
            elif diagonal_range.size:
                for coord in diagonal_range[:-1]:
                    if not self.is_position_empty(coord):
                        return False
                # If there is an ally figure at final_pos, False
                if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                    return False
                # If code execution reaches this point, the movement is valid, ergo True
                # If there is an enemy figure at final_pos, kill
                elif ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                    self.kill = True
                return True
            # Else (not a valid movement for Queen)
            return False

        # King
        elif figure == 6:
            # If trying to move one straight or diagonal step
            if straight_range.shape[0] == 1 or diagonal_range.shape[0] == 1:
                # If there is an ally figure at final_pos, False
                if ChessLogic.figure_for_player_at_position(self.board, self.current_player, final_pos) != 0:
                    return False
                else:
                    # If there is an enemy figure at final_pos, kill
                    if ChessLogic.figure_for_player_at_position(self.board, not self.current_player, final_pos) != 0:
                        self.kill = True
                    # Movement is valid, but player cannot castle anymore
                    if self.current_player == 1:
                        self.white_can_small_castle = False
                        self.white_can_large_castle = False
                    else:
                        self.black_can_small_castle = False
                        self.black_can_large_castle = False
                    return True
            # Else if trying to large castle
            elif ((np.array_equal(final_pos, [0, 2]) and self.current_player == 0) or
                  (np.array_equal(final_pos, [7, 2]) and self.current_player == 1)):
                # If player cannot castle, False
                if self.current_player == 1 and not self.white_can_large_castle:
                    return False
                # If player cannot castle, False
                if self.current_player == 0 and not self.black_can_large_castle:
                    return False

                # If figures in the way, return False
                for coord in straight_range:
                    if not self.is_position_empty(coord):
                        return False
                self.castle = 2
                return True
            # Else if trying to small castle
            elif ((np.array_equal(final_pos, [0, 6]) and self.current_player == 0) or
                  (np.array_equal(final_pos, [7, 6]) and self.current_player == 1)):
                # If player cannot castle, False
                if self.current_player == 1 and not self.white_can_small_castle:
                    return False
                # If player cannot castle, False
                if self.current_player == 0 and not self.black_can_small_castle:
                    return False

                # If figures in the way, return False
                for coord in straight_range:
                    if not self.is_position_empty(coord):
                        return False
                self.castle = 1
                return True
            return False

    def is_position_empty(self, coord):
        """
        Given a coordinate, returns True if the position is empty for both players, False otherwise
        :param coord: (1x2 tuple) Coordinate to check if empty
        :return: (bool) True if its empty, False otherwise
        """
        white_figure = ChessLogic.figure_for_player_at_position(self.board, 1, coord)
        black_figure = ChessLogic.figure_for_player_at_position(self.board, 0, coord)
        return white_figure == black_figure == 0

    @staticmethod
    def figure_for_player_at_position(board, player, coord):
        """
        Given a player and a coordinate, returns the figure.
        :param player: (int) The player
        :param coord: (1x2 tuple) The position in board
        :return:
        """
        position = board[int(player)].T[coord[1]][coord[0]]
        return np.where(position == 1)[0][0]

    def will_player_be_in_check_after_removing_figure_at_location_to_location(self, ori, fin):
        # First, assure ori and fin are tuples
        ori = tuple(ori)
        fin = tuple(fin)

        king_board = self.board[int(self.current_player)][6]
        king_where = np.where(king_board == 1)
        king_coord = (king_where[0][0], king_where[1][0])

        # Apply changes on a temporary board
        temp_board = np.copy(self.board)
        moving_figure = ChessLogic.figure_for_player_at_position(self.board, self.current_player, ori)
        temp_board[int(self.current_player)][moving_figure][ori] = 0
        temp_board[int(self.current_player)][moving_figure][fin] = 1
        temp_board[int(self.current_player)][0][ori] = 1
        temp_board[int(self.current_player)][0][fin] = 0

        # If it implies a kill
        enemy_figure = ChessLogic.figure_for_player_at_position(self.board, not self.current_player, fin)
        if enemy_figure != 0:
            temp_board[int(not self.current_player)][enemy_figure][fin] = 0
            temp_board[int(not self.current_player)][0][fin] = 1

        diagonals = ChessLogic.diagonal_coordinate_range(king_coord, ori)
        straights = ChessLogic.straight_coordinate_range(king_coord, ori)

        if diagonals.size:
            killing_figures = [4, 5]
            if ori[0] < king_coord[0] and ori[1] < king_coord[1]:
                diag = np.array(ori) - min(ori)
            elif ori[0] < king_coord[0] and ori[1] > king_coord[1]:
                diag = (ori[0], ori[1])
                while diag[0] > 0 and diag[1] < 7:
                    diag = (diag[0] - 1, diag[1] + 1)
            elif ori[0] > king_coord[0] and ori[1] < king_coord[1]:
                diag = (ori[0], ori[1])
                while diag[0] < 7 and diag[1] > 0:
                    diag = (diag[0] + 1, diag[1] - 1)
            else:
                diag = np.array(ori) + (7 - max(ori))
            all_diagonals = ChessLogic.diagonal_coordinate_range(king_coord, diag)
            for d in all_diagonals:
                ally_fig = np.where(temp_board[int(self.current_player)].T[d[1]][d[0]] == 1)[0][0]
                if ally_fig != 0:
                    return False
                enemy_fig = np.where(temp_board[int(not self.current_player)].T[d[1]][d[0]] == 1)[0][0]
                if enemy_fig in killing_figures:
                    return True
                elif enemy_fig != 0:
                    return False

        elif straights.size:
            killing_figures = [2, 5]
            if ori[0] < king_coord[0]:
                line = (0, king_coord[1])
            elif ori[0] > king_coord[0]:
                line = (7, king_coord[1])
            elif ori[1] < king_coord[1]:
                line = (king_coord[0], 0)
            else:
                line = (king_coord[0], 7)
            all_lines = ChessLogic.straight_coordinate_range(king_coord, line)
            for l in all_lines:
                if np.where(temp_board[int(self.current_player)].T[l[1]][l[0]] == 1)[0][0] != 0:
                    return False
                f = np.where(temp_board[int(not self.current_player)].T[l[1]][l[0]] == 1)[0][0]
                if f in killing_figures:
                    return True
                elif f != 0:
                    return False

        return False

    @staticmethod
    def straight_coordinate_range(initial_pos, final_pos):
        """
        Given an initial 2d tuple and a final 2d tuple, return all straight coordinates between both.
        :param initial_pos: (1x2 tuple) Initial position
        :param final_pos: (1x2 tuple) Final position
        :return: (np.array) of coordinates
        """

        delta_x = final_pos[0] - initial_pos[0]
        delta_y = final_pos[1] - initial_pos[1]

        # If not a straight coords, False
        if delta_x == delta_y or (delta_x != 0 and delta_y != 0):
            return np.array([])

        # If straight in x
        elif delta_x == 0:
            return np.array([[initial_pos[0], x] for x in range(initial_pos[1] + np.sign(delta_y),
                                                                final_pos[1] + np.sign(delta_y),
                                                                np.sign(delta_y))])
        # Else if straight in y)
        elif delta_y == 0:
            return np.array([[x, initial_pos[1]] for x in range(initial_pos[0] + np.sign(delta_x),
                                                                final_pos[0] + np.sign(delta_x),
                                                                np.sign(delta_x))])

    @staticmethod
    def diagonal_coordinate_range(initial_pos, final_pos):
        """
        Given an initial 2d tuple and a final 2d tuple, return all diagonal coordinates between both.
        :param initial_pos: (1x2 tuple) Initial position
        :param final_pos: (1x2 tuple) Final position
        :return: (np.array) of coordinates
        """
        delta_x = final_pos[0] - initial_pos[0]
        delta_y = final_pos[1] - initial_pos[1]

        # If not diagonal coords, False
        if abs(delta_x) != abs(delta_y) or (delta_x == 0 and delta_y == 0):
            return np.array([])

        return np.array(zip([x for x in range(initial_pos[0] + np.sign(delta_x),
                                              final_pos[0] + np.sign(delta_x),
                                              np.sign(delta_x))],
                            [x for x in range(initial_pos[1] + np.sign(delta_y),
                                              final_pos[1] + np.sign(delta_y),
                                              np.sign(delta_y))]))

    @staticmethod
    def are_coordinates_straight(coord_a, coord_b):
        if coord_a[0] == coord_b[0] or coord_a[1] == coord_b[1]:
            return True

        return False

    @staticmethod
    def are_coordinates_diagonal(coord_a, coord_b):
        delta_x = coord_b[0] - coord_a[0]
        delta_y = coord_b[1] - coord_a[1]

        if abs(delta_x) != abs(delta_y) or (delta_x == 0 and delta_y == 0):
            return False

        return True

    @staticmethod
    def valid_knight_positions_for_knight_at_location(coord):
        up_right = (coord[0] - 2, coord[1] + 1)
        right_up = (coord[0] - 1, coord[1] + 2)
        right_down = (coord[0] + 1, coord[1] + 2)
        down_right = (coord[0] + 2, coord[1] + 1)
        down_left = (coord[0] + 2, coord[1] - 1)
        left_down = (coord[0] + 1, coord[1] - 2)
        left_up = (coord[0] - 1, coord[1] - 2)
        up_left = (coord[0] - 2, coord[1] - 1)

        possible_positions = [up_right, right_up, right_down, down_right, down_left, left_down, left_up, up_left]
        valid_positions = []

        for position in possible_positions:
            if not (np.array(position) < 0).any() and not (np.array(position) > 7).any():
                valid_positions.append(position)

        return valid_positions

    def valid_bishop_positions_for_bishop_at_location(self, coord):
        valid_coordinates = []
        np_coord = np.array(coord)

        # Upper Left
        ul = ChessLogic.diagonal_coordinate_range(coord, np_coord - min(coord))
        for pos in ul:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break

        # Lower Right
        lr = ChessLogic.diagonal_coordinate_range(coord, np_coord + (7 - max(np_coord)))
        for pos in lr:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break

        # Upper Right
        x = coord[1] + 1
        y = coord[0] - 1
        while x < 8 and y >= 0:
            valid_coordinates.append(np.array((y, x)))
            if not self.is_position_empty((y, x)):
                break
            x += 1
            y -= 1

        # Lower Left
        x = coord[1] - 1
        y = coord[0] + 1
        while x >= 0 and y < 8:
            valid_coordinates.append(np.array((y, x)))
            if not self.is_position_empty((y, x)):
                break
            x -= 1
            y += 1

        return valid_coordinates

    @staticmethod
    def valid_diagonal_pawn_positions_for_pawn_at_location(coord):
        valid_coordiantes = []

        # Upper Left
        y = coord[0] - 1
        x = coord[1] - 1
        if 0 <= y < 8 and 0 <= x < 8:
            valid_coordiantes.append((y, x))

        # Upper Right
        y = coord[0] - 1
        x = coord[1] + 1
        if 0 <= y < 8 and 0 <= x < 8:
            valid_coordiantes.append((y, x))

        # Lower Right
        y = coord[0] + 1
        x = coord[1] + 1
        if 0 <= y < 8 and 0 <= x < 8:
            valid_coordiantes.append((y, x))

        # Lower Left
        y = coord[0] + 1
        x = coord[1] - 1
        if 0 <= y < 8 and 0 <= x < 8:
            valid_coordiantes.append((y, x))

        return valid_coordiantes

    @staticmethod
    def valid_king_straight_positions_for_king_at_location(coord):
        valid_coordiantes = []

        # Up
        y = coord[0] - 1
        if y >= 0:
            valid_coordiantes.append((y, coord[1]))

        # Right
        x = coord[1] + 1
        if x < 8:
            valid_coordiantes.append((coord[0], x))

        # Down
        y = coord[0] + 1
        if y < 8:
            valid_coordiantes.append((y, coord[1]))

        # Left
        x = coord[1] - 1
        if x >= 0:
            valid_coordiantes.append((coord[0], x))

        return valid_coordiantes

    @staticmethod
    def valid_king_positions_for_king_at_location(coord):
        valid_coordiantes = []

        # Diagonals
        valid_coordiantes.extend(ChessLogic.valid_diagonal_pawn_positions_for_pawn_at_location(coord))

        # Straights
        valid_coordiantes.extend(ChessLogic.valid_king_straight_positions_for_king_at_location(coord))

        return valid_coordiantes

    def valid_queen_positions_for_queen_at_location(self, coord):
        valid_coordinates = []

        # Diagonals
        valid_coordinates.extend(self.valid_bishop_positions_for_bishop_at_location(coord))

        # Straights
        valid_coordinates.extend(self.valid_rook_positions_for_rook_at_location(coord))

        return valid_coordinates

    def valid_rook_positions_for_rook_at_location(self, coord):
        valid_coordinates = []

        # Left
        l_coords = ChessLogic.straight_coordinate_range(coord, (coord[0], 0))
        for pos in l_coords:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break
        # Right
        r_coords = ChessLogic.straight_coordinate_range(coord, (coord[0], 7))
        for pos in r_coords:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break
        # Up
        u_coords = ChessLogic.straight_coordinate_range(coord, (0, coord[1]))
        for pos in u_coords:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break
        # Down
        d_coords = ChessLogic.straight_coordinate_range(coord, (7, coord[1]))
        for pos in d_coords:
            valid_coordinates.append(pos)
            if not self.is_position_empty(pos):
                break

        return valid_coordinates

    # Methods to help ChessParser
    def columns_for_player_figure_at_row(self, player, figure, row):
        player_board = self.board[int(player)]
        figure_board = player_board[figure]
        return np.where(figure_board[row] == 1)[0]

    def rows_for_player_figure_at_column(self, player, figure, column):
        player_board = self.board[int(player)]
        figure_board = player_board[figure]
        return np.where(figure_board.T[column] == 1)[0]

    def __str__(self):
        """
        Printable representation of the board state
        :return: Printable representation of the board state
        """
        return ChessLogic.print_move(self.board)

    @staticmethod
    def print_move(board):
        figures_dict = {1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K'}

        printable_board = np.ndarray((8, 8), dtype="S1")
        printable_board[:] = " "

        white_board = board[1]
        black_board = board[0]

        # For every possible figure, locate it
        for i in range(1, 7):
            printable_board[np.where(white_board[i] == 1)] = figures_dict[i]
            printable_board[np.where(black_board[i] == 1)] = figures_dict[i].lower()

        return str(printable_board)
