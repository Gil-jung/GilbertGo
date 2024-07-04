import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo import goboard_fast as goboard
from dlgo.agent import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo.rl.experience import load_experience

__all__ = [
    'ValueAgent',
    'load_value_agent',
]


class ValueDataSet(Dataset):
    def __init__(self, experiencebuffers, transform=None):
        self.experiencebuffers = experiencebuffers
        self.transform = transform
        self.length = 0
        for buff in experiencebuffers:
            self.length += len(buff.rewards)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        div = idx // 1024
        mod = idx % 1024

        X = torch.tensor(self.experiencebuffers[div].states[mod], dtype=torch.float32)
        y = torch.tensor(self.experiencebuffers[div].rewards[mod], dtype=torch.float32)

        if self.transform:
            X = self.transform(X)

        return X, y


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
        return self.model(input_tensor.cuda()).cpu()

    def select_move(self, game_state):
        
        # Loop over all legal moves.
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
        board_tensors = torch.tensor(np.array(board_tensors), dtype=torch.float32)

        # Values of the next state from opponent's view.
        opp_values = self.predict(board_tensors)
        opp_values = torch.squeeze(opp_values, dim=1).detach().numpy()

        # Values from our point of view.
        values = opp_values * (-1)

        if self.policy == 'eps-greedy':
            ranked_moves = self.rank_moves_eps_greedy(values)
        elif self.policy == 'weighted':
            ranked_moves = self.rank_moves_weighted(values)

        for move_idx in ranked_moves:
            move = moves[move_idx]
            if not is_point_an_eye(game_state.board, move.point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=self.encoder.encode_point(move.point),
                    )
                self.last_move_value = float(values[move_idx])
                return move
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()
    
    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        # This ranks the moves from worst to best.
        ranked_moves = np.argsort(values)
        # Return them in best-to-worst order.
        return ranked_moves[::-1]
    
    def rank_moves_weighted(self, values):
        p = values / np.sum(values)
        p = np.power(p, 1.0 / self.temperature)
        p = p / np.sum(p)
        return np.random.choice(np.arange(0, len(values)), size=len(values), p=p, replace=False)
    
    def train(self, lr=0.1, batch_size=128):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        version = 0
        base = path + f"\\buffers\\winning_experiences_value_*.pt"
        files_num = len(glob.glob(base))
        pivot = int(files_num * 0.8)

        winning_train_buffers = []
        losing_train_buffers = []
        winning_test_buffers = []
        losing_test_buffers = []
        train_data_counts = 0
        
        for idx in range(files_num):
            if idx < pivot:
                experience_buffer = load_experience(result="winning", type="value", no=f'{idx}')
                train_data_counts += len(experience_buffer.rewards)
                winning_train_buffers.append(experience_buffer)
                experience_buffer = load_experience(result="losing", type="value", no=f'{idx}')
                train_data_counts += len(experience_buffer.rewards)
                losing_train_buffers.append(experience_buffer)
            else:
                experience_buffer = load_experience(result="winning", type="value", no=f'{idx}')
                winning_test_buffers.append(experience_buffer)
                experience_buffer = load_experience(result="losing", type="value", no=f'{idx}')
                losing_test_buffers.append(experience_buffer)

        winning_train_dataset = ValueDataSet(winning_train_buffers, transform=trans_board)
        losing_train_dataset = ValueDataSet(losing_train_buffers, transform=trans_board)
        winning_test_dataset = ValueDataSet(winning_test_buffers, transform=trans_board)
        losing_test_dataset = ValueDataSet(losing_test_buffers, transform=trans_board)

        winning_train_loader = DataLoader(winning_train_dataset, batch_size=batch_size, shuffle=True)
        losing_train_loader = DataLoader(losing_train_dataset, batch_size=batch_size, shuffle=True)
        winning_test_loader = DataLoader(winning_test_dataset, batch_size=batch_size, shuffle=True)
        losing_test_loader = DataLoader(losing_test_dataset, batch_size=batch_size, shuffle=True)

        optimizer = SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        NUM_EPOCHES = 100
        self.model.cuda()
        total_steps = train_data_counts // batch_size * 8

        for epoch in range(NUM_EPOCHES):
            self.model.train()
            tot_loss = 0.0
            steps = 0

            for x, y in losing_train_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self.model(x)
                y_ = torch.squeeze(y_, dim=1)
                loss = loss_fn(y_, y.cuda()) 
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()

                if steps >= total_steps // 2:
                    break

            for x, y in winning_train_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self.model(x)
                y_ = torch.squeeze(y_, dim=1)
                loss = loss_fn(y_, y.cuda())
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()

                if steps >= total_steps:
                    break

            print('='*50)
            print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))
            # _, argmax = torch.max(y_, dim=1)
            # train_acc = compute_acc(argmax, y)
            # print("Epoch {}, Acc(train) : {}".format(epoch+1, train_acc))

            self.model.eval()
            eval_loss = 0
            x, y = next(iter(losing_test_loader))
            x = x.cuda()
            y_ = self.model(x)
            y_ = torch.squeeze(y_, dim=1)
            loss = loss_fn(y_, y.cuda())
            eval_loss += loss.item()
            x, y = next(iter(winning_test_loader))
            x = x.cuda()
            y_ = self.model(x)
            y_ = torch.squeeze(y_, dim=1)
            loss = loss_fn(y_, y.cuda()) 
            eval_loss += loss.item()
            eval_loss /= 2
            print("Epoch {}, Loss(val) : {}".format(epoch+1, eval_loss))
            # _, argmax = torch.max(y_, dim=1)
            # test_acc = compute_acc(argmax, y)
            # print("Epoch {}, Acc(val) : {}".format(epoch+1, test_acc))

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'loss': eval_loss,
            }, path + f"\\checkpoints\\alphago_rl_value_epoch_{epoch+1}_v{version}.pt")

        self.model.cpu()
    
    def serialize(self, version='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self.encoder.name(),
            'board_width': self.encoder.board_width,
            'board_height': self.encoder.board_height,
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
        }, path + f"\\agents\\AlphaGo_Value_Agent_{version}.pt")
    
    def diagnostics(self):
        return {'value': self.last_move_value}


def load_value_agent(version='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\AlphaGo_Value_Agent_{version}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = AlphaGoEncoder(use_player_plane=True)
    return ValueAgent(model, encoder)

def compute_acc(argmax, y):
        count = 0
        for i in range(len(argmax)):
            if argmax[i] == y[i]:
                count += 1
        return count / len(argmax)

def trans_board(state):
    val = random.randint(0, 7)
    if val == 0:
        return state
    elif val == 1:
        return torch.rot90(state, k=1, dims=[1, 2])
    elif val == 2:
        return torch.flip(state, dims=[1])
    elif val == 3:
        return torch.rot90(torch.flip(state, dims=[1]), k=1, dims=[1, 2])
    elif val == 4:
        return torch.flip(state, dims=[2])
    elif val == 5:
        return torch.rot90(torch.flip(state, dims=[2]), k=1, dims=[1, 2])
    elif val == 6:
        return torch.flip(torch.flip(state, dims=[2]), dims=[1])
    elif val == 7:
        return torch.rot90(torch.flip(torch.flip(state, dims=[2]), dims=[1]), k=1, dims=[1, 2])