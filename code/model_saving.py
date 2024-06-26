from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.networks.large import Large

import os
import torch

current_path = os.path.dirname(__file__)


go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols

encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
model = Large(go_board_rows, encoder.num_planes)
pt_flie = torch.load(current_path + "\\checkpoints\\alphago_sl_policy_epoch_7.pt")
model.load_state_dict(pt_flie['model_state_dict'])

agent = DeepLearningAgent(model, encoder)
agent.serialize()