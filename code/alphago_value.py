from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl import ValueAgent, load_experience


def main():
    rows, cols = 19, 19
    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()

    alphago_value = ValueAgent(model, encoder)

    winning_exp_buffer, losing_exp_buffer = load_experience(name='0001')

    alphago_value.train(
        winning_exp_buffer, losing_exp_buffer,
        lr=0.001,
        batch_size=128,
    )

    alphago_value.serialize(version='v0')


if __name__ == '__main__':
    main()