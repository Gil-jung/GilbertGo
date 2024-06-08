from dlgo.agent.naive import RandomBot
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
import time


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: RandomBot(),
        gotypes.Player.white: RandomBot(),
    }
    while not game.is_over():
        time.sleep(0.3)  # We set a sleep timer to 0.3 seconds so that bot moves aren't printed too fast to observe

        print(chr(27) + "[2J")  # Before each move we clear the screen. This way the board is always printed to the same position on the command line.
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()