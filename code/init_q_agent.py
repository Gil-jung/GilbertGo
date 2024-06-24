import argparse
import torch.nn as nn

from dlgo import rl
from dlgo import encoders
from dlgo.networks.large_q import Large_Q


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('output_file')
    args = parser.parse_args()
    board_size = args.board_size
    hidden_size = args.hidden_size
    output_file = args.output_file

    encoder = encoders.SimpleEncoder((board_size, board_size))
    model = Large_Q(board_size, encoder.num_planes, hidden_size)
    new_agent = rl.QAgent(model, encoder)
    new_agent.model.apply(initialize_weights)

    new_agent.serialize(output_file)


if __name__ == '__main__':
    main()