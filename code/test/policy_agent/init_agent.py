import argparse
import torch.nn as nn

from dlgo import agent
from dlgo import encoders
from dlgo.networks.large import Large


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
    parser.add_argument('output_file')
    args = parser.parse_args()
    board_size = args.board_size
    output_file = args.output_file

    encoder = encoders.SimpleEncoder((board_size, board_size))
    model = Large(board_size, encoder.num_planes)
    new_agent = agent.PolicyAgent(model, encoder)
    new_agent._model.apply(initialize_weights)

    new_agent.serialize(output_file)


if __name__ == '__main__':
    main()