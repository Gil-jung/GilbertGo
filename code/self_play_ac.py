import os
import argparse
import datetime
from collections import namedtuple

from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}
path = os.path.dirname(__file__)


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


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


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(black_player.model.img_size)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--game-log-out', required=True)
    parser.add_argument('--experience-out', required=True)
    # parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()
    learning_agent = args.learning_agent
    num_games = args.num_games
    game_log_out = args.game_log_out
    experience_out = args.experience_out
    # temperature = args.temperature

    agent1 = rl.load_ac_agent(name=learning_agent)
    agent2 = rl.load_ac_agent(name=learning_agent)
    # agent1.set_temperature(temperature)
    # agent2.set_temperature(temperature)
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    logf = open(path + "\\logs\\" + game_log_out + ".txt", 'a')
    logf.write('Begin training at %s\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),))
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        collector2.begin_episode()
        
        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.black:
            print('Agent1 wins.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Agent2 wins.')
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
    
    winning_experiences, losing_experiences = rl.combine_experience([collector1, collector2])
    logf.write('Saving experience buffer to %s\n' % experience_out)
    winning_experiences.serialize(result='winning', name=experience_out)
    losing_experiences.serialize(result='losing', name=experience_out)


if __name__ == '__main__':
    main()
