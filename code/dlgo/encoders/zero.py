import numpy as np

from dlgo.goboard_fast  import Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import Encoder


class ZeroEncoder(Encoder):
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.num_planes = 17
    
    def name(self):
        return 'zero'
    
    def encode(self, game_state):
        board_tensor = np.zeros((self.num_planes, self.board_size, self.board_size))
        for r in range(self.board_size):
            for c in range(self.board_size):
                point = Point(row=r + 1, col=c + 1)

                go_string = game_state.board.get_go_string(point)
                if go_string and go_string.color == game_state.next_player:
                    for i in range(8):
                        board_tensor[i*2][r][c] = 1
                elif go_string and go_string.color == game_state.next_player.other:
                    for i in range(8):
                        board_tensor[i*2 + 1][r][c] = 1

                age = game_state.board.move_ages.get(r, c)
                if age >= 0 and age < 7 and go_string.color == game_state.next_player:
                    for i in range(age, 7):
                        board_tensor[i*2 + 2][r][c] = 0
                if age >= 0 and age < 7 and go_string.color == game_state.next_player.other:
                    for i in range(age, 7):
                        board_tensor[i*2 + 3][r][c] = 0
        
        if game_state.next_player == Player.black:
            board_tensor[16] = np.ones((1, self.board_size, self.board_size))
        else:
            board_tensor[16] = np.zeros((1, self.board_size, self.board_size))

        return board_tensor
    
    def encode_move(self, move):
        if move.is_play:
            return (self.board_size * (move.point.row - 1) + (move.point.col - 1))
        elif move.is_pass:
            return self.board_size * self.board_size
        raise ValueError('Cannot encode resign move')
    
    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))
    
    def num_moves(self):
        return self.board_size * self.board_size + 1
    
    def shape(self):
        return self.num_planes, self.board_size, self.board_size

def create(board_size):
    return ZeroEncoder(board_size)