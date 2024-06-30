from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.rl.simulate import experience_simulation
from dlgo.rl.value import ValueAgent


def main():
    alphago_rl_agent = load_policy_agent(type='SL', version='v0')
    opponent = load_policy_agent(type='SL', version='v0')
    alphago_rl_agent._model.cuda()
    opponent._model.cuda()

    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()
    value_agent1 = ValueAgent(model, encoder)
    value_agent2 = ValueAgent(model, encoder)

    num_games = 10
    policy_exp_buffer, value_exp_buffer = experience_simulation(
        num_games, alphago_rl_agent, opponent, value_agent1, value_agent2
    )

    alphago_rl_agent.train(
        policy_exp_buffer[0], policy_exp_buffer[1],
        lr=0.001,
        clipnorm=1.0,
        batch_size=128
    )

    alphago_rl_agent.serialize(type='RL', version='test')

    value_exp_buffer[0].serialize(result='winning', name='test')
    value_exp_buffer[1].serialize(result='losing', name='test')


if __name__ == '__main__':
    main()