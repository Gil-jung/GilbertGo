import argparse
import numpy as np

from dlgo.encoders import get_encoder_by_name
from dlgo import goboard_fast as goboard
from dlgo import mcts
from dlgo.utils import print_board, print_move


def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []  # In `boards` we store encoded board state, `moves` is for encoded moves.

    encoder = get_encoder_by_name('oneplane', board_size)  # We initialize a OnePlaneEncoder by name with given board size.

    game = goboard.GameState.new_game(board_size)  # An new game of size `board_size` is instantiated.

    bot = mcts.MCTSAgent(rounds, temperature)  # A Monte Carlo tree search agent with specified number of rounds and temperature will serve as our bot.

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)  # The next move is selected by the bot.
        if move.is_play:
            boards.append(encoder.encode(game))  # The encoded board situation is appended to `boards`.

            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)  # The one-hot-encoded next move is appended to `moves`.

        print_move(game.next_player, move)
        game = game.apply_move(move)  # Afterwards the bot move is applied to the board.
        num_moves += 1
        if num_moves > max_moves:  # We continue with the next move, unless the maximum number of moves has been reached.
            break

    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    args = parser.parse_args()  # This application allows some customization via command line arguments.
    xs = []
    ys = []

    for i in range(args.num_games):
        print('Generating game %d/%d...' % (i + 1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)  # For the specified number of games we generate game data.
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)  # After all games have been generated, we concatenate features and labels, respectively.
    y = np.concatenate(ys)

    np.save(args.board_out, x)  # We store feature and label data to separate files, as specified by the command line options.
    np.save(args.move_out, y)


if __name__ == '__main__':
    main()