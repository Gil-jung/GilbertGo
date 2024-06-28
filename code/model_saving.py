from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoPolicyResNet

import os
import torch

current_path = os.path.dirname(__file__)
version = 'v1'
saving_epoch = 32

encoder = AlphaGoEncoder(use_player_plane=False)
model = AlphaGoPolicyResNet()
pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_sl_policy_epoch_{saving_epoch}_{version}.pt")
model.load_state_dict(pt_flie['model_state_dict'])

agent = PolicyAgent(model, encoder)
agent.serialize(type='SL', version='v1')