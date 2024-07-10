from dlgo.agent.pg import PolicyAgent
from dlgo.rl.value import ValueAgent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoPolicyResNet, AlphaGoValueResNet

import os
import torch

current_path = os.path.dirname(__file__)
type = 'RL'
version = 'v0'
saving_epoch = 61

##############################################################################################################

encoder = AlphaGoEncoder(use_player_plane=False)
model = AlphaGoPolicyResNet()
pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_{type}_policy_epoch_{saving_epoch}_{version}.pt")
model.load_state_dict(pt_flie['model_state_dict'])

agent = PolicyAgent(model, encoder)
agent.serialize(type=type, version=version)

##############################################################################################################

# encoder = AlphaGoEncoder(use_player_plane=True)
# model = AlphaGoValueResNet()
# pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_{type}_value_epoch_{saving_epoch}_{version}.pt")
# model.load_state_dict(pt_flie['model_state_dict'])

# agent = ValueAgent(model, encoder)
# agent.serialize(version=version)