from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.rl.simulate import experience_simulation
from dlgo.rl.value import ValueAgent
from dlgo.rl.experience import ExperienceBuffer, load_experience

import numpy as np


def main():
    alphago_rl_agent = load_policy_agent(type='SL', version='v0')
    opponent = load_policy_agent(type='SL', version='v0')
    alphago_rl_agent._model.cuda()
    opponent._model.cuda()

    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()
    value_agent1 = ValueAgent(model, encoder)
    value_agent2 = ValueAgent(model, encoder)

    num_games = 6
    policy_exp_buffer, value_exp_buffer = experience_simulation(
        num_games, alphago_rl_agent, opponent, value_agent1, value_agent2
    )

    policy_exp_buffer[0].serialize(result='winning', name='policy_test')
    policy_exp_buffer[1].serialize(result='losing', name='policy_test')
    value_exp_buffer[0].serialize(result='winning', name='value_test')
    value_exp_buffer[1].serialize(result='losing', name='value_test')

    policy_winning_buffer, policy_losing_buffer = load_experience(name='policy_test')

    alphago_rl_agent.train(
        policy_winning_buffer, policy_losing_buffer,
        lr=0.001,
        clipnorm=1.0,
        batch_size=128
    )

    alphago_rl_agent.serialize(type='RL', version='v0')


if __name__ == '__main__':
    main()