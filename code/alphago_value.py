from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl.value import ValueAgent
import os
import torch
import torch.nn as nn


def main():
    path = os.path.dirname(__file__)
    pre_trained = False
    version = 0
    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()
    
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

    if not pre_trained:
        model.apply(initialize_weights)
        print("initializing...")
    else:
        pt_flie = torch.load(path + f"\\agents\\AlphaGo_Value_Agent_v{version}.pt")
        model.load_state_dict(pt_flie['model_state_dict'])
        print("model loading...")

    alphago_value = ValueAgent(model, encoder)

    alphago_value.train(
        lr=0.001,
        batch_size=128,
    )

    alphago_value.serialize(version='v0')


if __name__ == '__main__':
    main()