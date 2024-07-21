from dlgo.agent.pg import PolicyAgent
from dlgo.rl.value import ValueAgent
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks.alphago import AlphaGoPolicyMiniResNet, AlphaGoValueMiniResNet
from dlgo.encoders.zero import ZeroEncoder
from dlgo.networks.alphago_zero import AlphaGoZeroMiniNet
from dlgo.zero.agent import ZeroAgent

import os
import torch

current_path = os.path.dirname(__file__)
type = 'RL'
version = 'v0'
saving_epoch = 87

##############################################################################################################

# encoder = SimpleEncoder(board_size=(19, 19))
# model = AlphaGoPolicyMiniResNet(num_planes=11)
# pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_{type}_policy_epoch_{saving_epoch}_{version}.pt")
# model.load_state_dict(pt_flie['model_state_dict'])

# agent = PolicyAgent(model, encoder)
# agent.serialize(type=type, version=version)

##############################################################################################################

encoder = SimpleEncoder(board_size=(19, 19))
model = AlphaGoValueMiniResNet(num_planes=11)
pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_{type}_value_epoch_{saving_epoch}_{version}.pt")
model.load_state_dict(pt_flie['model_state_dict'])

agent = ValueAgent(model, encoder)
agent.serialize(version=version)

##############################################################################################################

# encoder = ZeroEncoder()
# model = AlphaGoZeroMiniNet()
# pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_{type}_zero_epoch_{saving_epoch}_{version}.pt")
# model.load_state_dict(pt_flie['model_state_dict'])

# agent = ZeroAgent(model, encoder)
# agent.serialize(version=version)