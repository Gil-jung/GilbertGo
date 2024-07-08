from dlgo import scoring
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.encoders.zero import ZeroEncoder
from dlgo.networks.alphago_zero import AlphaGoZeroMiniNet
from dlgo.zero.agent import ZeroAgent, load_zero_agent
from dlgo.zero.experience import ZeroExperienceBuffer, ZeroExperienceCollector, combine_experience

from collections import namedtuple

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def simulate_game(board_size, black_agent, white_agent):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)

    return game_result


def main():
    pre_trained = False
    version = 'v0'
    num_games = 2
    board_size = 19

    if pre_trained == False:
        encoder = ZeroEncoder()
        model = AlphaGoZeroMiniNet()
        agent1 = ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
        agent2 = ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    else:
        agent1 = load_zero_agent(version)
        agent2 = load_zero_agent(version)
    
    agent1.model.cuda()
    agent2.model.cuda()
    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    agent1.set_collector(c1)
    agent2.set_collector(c2)
    
    base_color = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        
        if base_color == Player.black:
            game_record = simulate_game(board_size, agent1, agent2)
        else:
            game_record = simulate_game(board_size, agent2, agent1)

        print(game_record)
        if game_record.winner == base_color:
            c1.complete_episode(reward=1)
            c2.complete_episode(reward=-1)
        else:
            c2.complete_episode(reward=1)
            c1.complete_episode(reward=-1)
        base_color = base_color.other
    
    exp_buffer = combine_experience([c1, c2])

    winning_states = exp_buffer[0].states
    winning_visit_counts = exp_buffer[0].visit_counts
    winning_advantages = exp_buffer[0].advantages
    losing_states = exp_buffer[1].states
    losing_visit_counts = exp_buffer[1].visit_counts
    losing_advantages = exp_buffer[1].advantages
    
    chunk = 0
    chunk_size = 1024
    while len(winning_advantages) >= chunk_size:
        current_winning_states, winning_states = winning_states[:chunk_size], winning_states[chunk_size:]
        current_winning_visit_counts, winning_visit_counts = winning_visit_counts[:chunk_size], winning_visit_counts[chunk_size:]
        current_winning_advantages, winning_advantages = winning_advantages[:chunk_size], winning_advantages[chunk_size:]
        current_losing_states, losing_states = losing_states[:chunk_size], losing_states[chunk_size:]
        current_losing_visit_counts, losing_visit_counts = losing_visit_counts[:chunk_size], losing_visit_counts[chunk_size:]
        current_losing_advantages, losing_advantages = losing_advantages[:chunk_size], losing_advantages[chunk_size:]

        ZeroExperienceBuffer(
            current_winning_states,
            current_winning_visit_counts,
            [],
            current_winning_advantages
        ).serialize(result="winning", name=f'zero_{chunk}')
        ZeroExperienceBuffer(
            current_losing_states,
            current_losing_visit_counts,
            [],
            current_losing_advantages
        ).serialize(result="losing", name=f'zero_{chunk}')

        chunk += 1

    ZeroExperienceBuffer(
        winning_states,
        winning_visit_counts,
        [],
        winning_advantages
    ).serialize(result="winning", name=f'zero_{chunk}')
    ZeroExperienceBuffer(
        losing_states,
        losing_visit_counts,
        [],
        losing_advantages
    ).serialize(result="losing", name=f'zero_{chunk}')

    agent1.train(
        learning_rate=0.001,
        clipnorm=1.5,
        batch_size=128
    )

    agent1.serialize(type='RL', version=version)


if __name__ == '__main__':
    main()