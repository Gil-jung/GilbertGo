import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo.agent.base import Agent
from dlgo.agent.helper_fast import is_point_an_eye
from dlgo import encoders
from dlgo import goboard_fast as goboard

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


class ACDataSet(Dataset):
    def __init__(self, experience, transform=None):
        self.experience = experience
        self.transform = transform
    
    def __len__(self):
        return len(self.experience.states)
    
    def __getitem__(self, idx):
        states = torch.tensor(self.experience.states, dtype=torch.float32)[idx]
        actions = torch.tensor(self.experience.actions, dtype=torch.float32)[idx]
        rewards = torch.tensor(self.experience.rewards, dtype=torch.float32)[idx]
        advantages = torch.tensor(self.experience.advantages, dtype=torch.float32)[idx]

        if self.transform:
            states = self.transform(states)

        return states, (actions, rewards, advantages)


class ACAgent(Agent):
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None

        self.last_state_value = 0
    
    def set_collector(self, collector):
        self.collector = collector

    def predict(self, input_tensor):
        return self.model(input_tensor)

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        input_tensor = torch.unsqueeze(torch.tensor(board_tensor, dtype=torch.float32), dim=0)

        actions, values = self.predict(input_tensor)
        move_probs = torch.squeeze(actions, dim=0).detach().numpy()
        estimated_value = torch.squeeze(values, dim=0).detach().numpy()

        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(
                game_state.board, point, game_state.next_player
            )
            if move_is_valid and (not fills_own_eye):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value[0]
                    )
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()
    
    def train(self, winning_exp_buffer, losing_exp_buffer, lr=0.1, batch_size=128):
        winning_exp_dataset = ACDataSet(winning_exp_buffer)
        winning_exp_loader = DataLoader(winning_exp_dataset, batch_size=batch_size)
        losing_exp_dataset = ACDataSet(losing_exp_buffer)
        losing_exp_loader = DataLoader(losing_exp_dataset, batch_size=batch_size)
        optimizer = SGD(self.model.parameters(), lr=lr)
        policy_loss_fn = CELoss
        value_loss_fn = nn.MSELoss()
        NUM_EPOCHES = 5
        self.model.cuda()

        for epoch in range(NUM_EPOCHES):
            self.model.train()
            tot_loss = 0.0
            steps = 0

            for x, y in winning_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self.model(x)
                loss = policy_loss_fn(y_[0], y[0].cuda(), y[2].cuda()) + value_loss_fn(y_[1], y[1].cuda()) * 0.5
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()

            for x, y in losing_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self.model(x)
                loss = 1 - policy_loss_fn(y_[0], y[0].cuda(), y[2].cuda()) + value_loss_fn(y_[1], y[1].cuda()) * 0.5
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()

            print('='*100)
            print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))

        self.model.cpu()
    
    def serialize(self, name='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self.encoder.name(),
            'board_width': self.encoder.board_width,
            'board_height': self.encoder.board_height,
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
        }, path + f"\\agents\\AC_Agent_{self.model.name()}_{self.encoder.name()}_{name}.pt")


def load_ac_agent(model_name='large_ac', encoder_name='simple', name='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\AC_Agent_{model_name}_{encoder_name}_{name}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height)
    )
    return ACAgent(model, encoder)


def CELoss(output, action, advantage):
    size = len(output)
    result = torch.zeros(size)
    for i in range(size):
        value = (-1.0)*torch.log(torch.exp(output[i][action[i].long()]) / torch.sum(torch.exp(output[i])))*advantage[i]
        result[i] = value
    return torch.mean(result)