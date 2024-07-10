import argparse
import multiprocessing
import os, time, random
import numpy as np
from collections import namedtuple

from dlgo import agent
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.agent.pg import load_policy_agent


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player, board_size=19):
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
    
    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size = args

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_policy_agent(type='SL', version=agent1_fname)
    agent2 = load_policy_agent(type='SL', version=agent2_fname)

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', '-a1', required=True)
    parser.add_argument('--agent2', '-a2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-size', '-b', type=int, default=19)

    args = parser.parse_args()

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_policy_agent(type='RL', version=args.agent1)
    agent2 = load_policy_agent(type='SL', version=args.agent2)
    agent1._model.cuda()
    agent2._model.cuda()

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(args.num_games):
        print('Simulating game %d/%d...' % (i + 1, args.num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, args.board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other


if __name__ == '__main__':
    main()