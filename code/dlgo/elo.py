import random

import numpy as np
from scipy.optimize import minimize

from dlgo import scoring
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player

__all__ = [
    'calculate_ratings',
]


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)

    return game_result.winner


def nll_results(ratings, winners, losers):
    all_ratings = np.concatenate()