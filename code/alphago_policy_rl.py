from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.rl.simulate import experience_simulation
from dlgo.rl.value import ValueAgent


def main():
    alphago_rl_agent = load_policy_agent(type='SL', version='v0')
    opponent = load_policy_agent(type='SL', version='v0')

    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()
    dummy_agent = ValueAgent(model, encoder)

    num_games = 5
    winning_experiences, losing_experiences = experience_simulation(num_games, alphago_rl_agent, opponent, dummy_agent, dummy_agent)

    alphago_rl_agent.train(
        winning_experiences, losing_experiences,
        lr=0.0001,
        clipnorm=1.0,
        batch_size=512
    )

    alphago_rl_agent.serialize(type='RL', version='v0')

    winning_experiences.serialize(result='winning', name='0001')
    losing_experiences.serialize(result='losing', name='0001')


if __name__ == '__main__':
    main()