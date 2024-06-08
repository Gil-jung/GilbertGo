import copy
from dlgo.gotypes import Player
from dlgo import zobrist

class GoString():  # Go strings are stones that are linked by a chain of connected stones of the same color.
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)  # `stones` and `liberties` are now immutable `frozenset` instances

    def without_liberty(self, point):  # The `without_liberty` methods replaces the previous `remove_liberty` method...
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string):  # Return a new Go string containing all stones in both strings.
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones
        )
    
    @property
    def num_liberties(self):
        return len(self.liberties)
    
    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties


class Board():  # A board is initialized as empty grid with the specified number of rows and columns.
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD
    
    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        if self._grid.get(point) is not None:
            print('Illegal play on %s' % str(point))
        assert self._grid.get(point) is None
        # Examine the adjacent points.
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
        for neighbor in point.neighbors():  # First, we examine direct neighbors of this point.
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)  # Until this line `place_stone` remains the same.
        
        for same_color_string in adjacent_same_color:  # You merge any adjacent strings of the same color.
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        self._hash ^= zobrist.HASH_CODE[point, player]  # Next, you apply the hash code for this point and player

        for other_color_string in adjacent_opposite_color:  # Reduce liberties of any adjacent strings of the opposite color.
            replacement = other_color_string.without_liberty(point)  # Then you reduce liberties of any adjacent strings of the opposite color.
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            else:
                self._remove_string(other_color_string)  # If any opposite color strings now have zero liberties, remove them.

    def _replace_string(self, new_string):  # This new helper method updates our Go board grid.
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():  # Removing a string can create liberties for other strings.
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            self._grid[point] = None

            self._hash ^= zobrist.HASH_CODE[point, string.color]  # With Zobrist hashing, you need to unapply the hash for this move.

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols
    
    def get(self, point):  # Returns the content of a point on the board:  a Player if there is a stone on that point or else None.
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color
    
    def get_go_string(self, point):  # Returns the entire string of stones at a point: a GoString if there is a stone on that point or else None.
        string = self._grid.get(point)
        if string is None:
            return None
        return string
    
    def zobrist_hash(self):
        return self._hash


class Move():  # Any action a player can play on a turn, either is_play, is_pass or is_resign will be set.
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign
    
    @classmethod
    def play(cls, point):  # This move places a stone on the board.
        return Move(point=point)
    
    @classmethod
    def pass_turn(cls):  # This move passes.
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):  # This move resigns the current game
        return Move(is_resign=True)


class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_state = frozenset()
        else:
            self.previous_state = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())}
            )
        self.last_move = move

    def apply_move(self, move):  # Return the new GameState after applying the move.
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)
    
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
    
    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0
    
    @property
    def situation(self):
        return (self.next_player, self.board)
    
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_state
    
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and 
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move)
        )
    
    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass