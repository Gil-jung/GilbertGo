from dlgo import rl
from dlgo import scoring
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Player

from collections import namedtuple

from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks.alphago import AlphaGoValueResNet, AlphaGoValueMiniResNet
from dlgo.rl.value import ValueAgent


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def simulate_game(black_player, white_player, black_value_agent, white_value_agent):
    moves = []
    game = goboard.GameState.new_game(19)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)

        if next_move != goboard.Move.pass_turn():
            if game.next_player == Player.black:
                black_value_agent.collector.record_decision(
                    state=black_value_agent.encoder.encode(game),
                    # action=black_value_agent.encoder.encode_point(next_move.point),
                )
            elif game.next_player == Player.white:
                white_value_agent.collector.record_decision(
                    state=white_value_agent.encoder.encode(game),
                    # action=white_value_agent.encoder.encode_point(next_move.point),
                )
                
        game = game.apply_move(next_move)
    
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def experience_simulation(num_games, agent1, agent2, agent3, agent4):
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    collector3 = rl.ExperienceCollector()
    collector4 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)
        collector3.begin_episode()
        agent3.set_collector(collector3)
        collector4.begin_episode()
        agent4.set_collector(collector4)

        if color1 == Player.black:
            # black_player, white_player = agent1, agent2
            game_record = simulate_game(agent1, agent2, agent3, agent4)
        else:
            # white_player, black_player = agent1, agent2
            game_record = simulate_game(agent2, agent1, agent4, agent3)
        # game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
            collector3.complete_episode(reward=1)
            collector4.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
            collector4.complete_episode(reward=1)
            collector3.complete_episode(reward=-1)
        color1 = color1.other

    policy_exp_buffer = rl.combine_experience([collector1, collector2])
    value_exp_buffer = rl.combine_experience([collector3, collector4])
    
    return policy_exp_buffer, value_exp_buffer


def experience_simulation_large(num_games):
    SL_version_1 = 'v7'
    SL_version_2_list = ['v7', 'v6', 'v5', 'v4', 'v3', 'v2', 'v1', 'v0']

    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    collector3 = rl.ExperienceCollector()
    collector4 = rl.ExperienceCollector()

    for SL_version_2 in SL_version_2_list:

        agent1 = load_policy_agent(type='SL', version=SL_version_1)
        agent2 = load_policy_agent(type='SL', version=SL_version_2)
        agent1._model.cuda()
        agent2._model.cuda()

        encoder = AlphaGoEncoder(use_player_plane=True)
        model = AlphaGoValueResNet()
        agent3 = ValueAgent(model, encoder)
        agent4 = ValueAgent(model, encoder)

        color1 = Player.black
        for i in range(num_games):
            print('Simulating game with %s : %d/%d...' % (SL_version_2, i + 1, num_games))
            collector1.begin_episode()
            agent1.set_collector(collector1)
            collector2.begin_episode()
            agent2.set_collector(collector2)
            collector3.begin_episode()
            agent3.set_collector(collector3)
            collector4.begin_episode()
            agent4.set_collector(collector4)

            if color1 == Player.black:
                # black_player, white_player = agent1, agent2
                game_record = simulate_game(agent1, agent2, agent3, agent4)
            else:
                # white_player, black_player = agent1, agent2
                game_record = simulate_game(agent2, agent1, agent4, agent3)
            # game_record = simulate_game(black_player, white_player)
            if game_record.winner == color1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
                collector3.complete_episode(reward=1)
                collector4.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
                collector4.complete_episode(reward=1)
                collector3.complete_episode(reward=-1)
            color1 = color1.other

    policy_exp_buffer = rl.combine_experience([collector1, collector2])
    value_exp_buffer = rl.combine_experience([collector3, collector4])
    
    return policy_exp_buffer, value_exp_buffer


def value_experience_simulation(num_games, agent1, agent2):
    SL_version_1 = agent1
    SL_version_2_list = agent2

    collector3 = rl.ExperienceCollector()
    collector4 = rl.ExperienceCollector()

    for SL_version_2 in SL_version_2_list:

        agent1 = load_policy_agent(type='SL', version=SL_version_1)
        agent2 = load_policy_agent(type='SL', version=SL_version_2)
        agent1._model.cuda()
        agent2._model.cuda()

        encoder = SimpleEncoder(board_size=(19, 19))
        model = AlphaGoValueMiniResNet(num_planes=11)
        agent3 = ValueAgent(model, encoder)
        agent4 = ValueAgent(model, encoder)

        color1 = Player.black
        for i in range(num_games):
            print('Simulating game with %s : %d/%d...' % (SL_version_2, i + 1, num_games))
            collector3.begin_episode()
            agent3.set_collector(collector3)
            collector4.begin_episode()
            agent4.set_collector(collector4)

            if color1 == Player.black:
                game_record = simulate_game(agent1, agent2, agent3, agent4)
            else:
                game_record = simulate_game(agent2, agent1, agent4, agent3)
            if game_record.winner == color1:
                collector3.complete_episode(reward=1)
                collector4.complete_episode(reward=-1)
            else:
                collector4.complete_episode(reward=1)
                collector3.complete_episode(reward=-1)
            color1 = color1.other

    value_exp_buffer = rl.combine_experience([collector3, collector4])
    
    return value_exp_buffer