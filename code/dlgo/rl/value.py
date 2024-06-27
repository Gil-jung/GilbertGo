import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo import encoders
from dlgo import goboard_fast as goboard
from dlgo.agent import Agent
from dlgo.agent.helpers_fast import is_point_an_eye

__all__ = [
    'ValueAgent',
    'load_value_agent',
]


class ValueDataSet(Dataset):
    def __init__(self, experience, num_moves=19*19, transform=None):
        self.experience = experience
        self.transform = transform
        self.num_moves = num_moves
    
    def __len__(self):
        return len(self.experience.states)
    
    def __getitem__(self, idx):
        states = torch.tensor(self.experience.states, dtype=torch.float32)[idx]
        actions = torch.zeros((len(self.experience.actions), self.num_moves), dtype=torch.float32)
        actions[idx][self.experience.actions[idx]] = 1
        actions = actions[idx]
        rewards = torch.zeros((len(self.experience.rewards),))
        rewards[idx] = 1 if self.experience.rewards[idx] > 0 else 0
        rewards = rewards[idx]

        if self.transform:
            (states, actions) = self.transform((states, actions))

        return (states, actions), rewards


class ValueAgent(Agent):
    def __init__(self, model, encoder, policy='eps-greedy'):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
        self.policy = policy

        self.last_move_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_collector(self, collector):
        self.collector = collector
    
    def set_policy(self, policy):
        if policy not in ('eps-greedy', 'weighted'):
            raise ValueError(policy)
        self.policy = policy
    
    def predict(self, input_tensor):
        return self.model(input_tensor)

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)

        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            next_state = game_state.apply_move(move)
            board_tensor = self.encoder.encode(next_state)
            moves.append(move)
            board_tensors.append(board_tensor)
        if not moves:
            return goboard.Move.pass_turn()

        # num_moves = len(moves)
        board_tensors = torch.tensor(board_tensors, dtype=torch.float32)



        values = self.predict(board_tensors)
        values = torch.squeeze(values, dim=1).detach().numpy()

        ranked_moves = self.rank_moves_eps_greedy(values)

        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=moves[move_idx],
                    )
                self.last_move_value = float(values[move_idx])
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()
    
    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        # This ranks the moves from worst to best.
        ranked_moves = np.argsort(values)
        # Return them in best-to-worst order.
        return ranked_moves[::-1]
    
    def train(self, winning_exp_buffer, losing_exp_buffer, lr=0.1, batch_size=128):
        winning_exp_dataset = QDataSet(winning_exp_buffer, num_moves=self.encoder.num_points())
        winning_exp_loader = DataLoader(winning_exp_dataset, batch_size=batch_size)
        losing_exp_dataset = QDataSet(losing_exp_buffer, num_moves=self.encoder.num_points())
        losing_exp_loader = DataLoader(losing_exp_dataset, batch_size=batch_size)
        optimizer = SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        NUM_EPOCHES = 5
        self.model.cuda()

        for epoch in range(NUM_EPOCHES):
            self.model.train()
            tot_loss = 0.0
            steps = 0

            for x, y in winning_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x[0], x[1] = x[0].cuda(), x[1].cuda()
                y_ = self.model(x)
                loss = loss_fn(y_, y.cuda()) 
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()

            for x, y in losing_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x[0], x[1] = x[0].cuda(), x[1].cuda()
                y_ = self.model(x)
                loss = loss_fn(y_, y.cuda())
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
        }, path + f"\\agents\\Q_Agent_{self.model.name()}_{self.encoder.name()}_{name}.pt")
    
    def diagnostics(self):
        return {'value': self.last_move_value}


def load_q_agent(model_name='large_q', encoder_name='simple', name='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\Q_Agent_{model_name}_{encoder_name}_{name}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height)
    )
    return QAgent(model, encoder)