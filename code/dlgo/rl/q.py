import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo import encoders
from dlgo import goboard_fast as goboard
from dlgo.agent import Agent
from dlgo.agent.helper_fast import is_point_an_eye

__all__ = [
    'QAgent',
    'load_q_agent',
]


class ExperienceDataSet(Dataset):
    def __init__(self, experience, transform=None):
        self.experience = experience
        self.transform = transform
    
    def __len__(self):
        return len(self.experience.states)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.experience.states, dtype=torch.float32)[idx]
        y = torch.tensor(self.experience.actions, dtype=torch.long)[idx]
        r = torch.tensor(self.experience.rewards, dtype=torch.long)[idx]

        if self.transform:
            X = self.transform(X)

        return X, (y, r)


class QAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

        self.last_move_value - 0

    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_collector(self, collector):
        self.collector = collector
    
    def predict(self, states, actions):
        return self.model(states, actions)

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)

        moves = []
        board_tensors =[]
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor)
        if not moves:
            return goboard.Move.pass_turn()
        
        num_moves = len(moves)
        board_tensors = torch.tensor(board_tensors, dtype=torch.float32)
        move_vectors = torch.zeros((num_moves, self.encoder.num_points()), dtype=torch.float32)
        for i, move in enumerate(moves):
            move_vectors[i][move] = 1
        
        values = self.predict(board_tensors, move_vectors)
        values = torch.squeeze(values, dim=0).detach().numpy()

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
    
    def train(self, winning_exp_buffer, losing_exp_buffer, lr=0.0001, clipnorm=1.0, batch_size=512):
        winning_exp_dataset = ExperienceDataSet(winning_exp_buffer)
        winning_exp_loader = DataLoader(winning_exp_dataset, batch_size=batch_size)
        losing_exp_dataset = ExperienceDataSet(losing_exp_buffer)
        losing_exp_loader = DataLoader(losing_exp_dataset, batch_size=batch_size)
        optimizer = SGD(self._model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        NUM_EPOCHES = 5
        self._model.cuda()

        for epoch in range(NUM_EPOCHES):
            self._model.train()
            tot_loss = 0.0
            steps = 0

            for x, y in winning_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self._model(x)
                loss = loss_fn(y_, y[0].cuda()) 
                loss.backward()
                tot_loss += loss.item()
                nn.utils.clip_grad_norm_(self._model.parameters(), clipnorm)
                optimizer.step()

            for x, y in losing_exp_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self._model(x)
                loss = 1 - loss_fn(y_, y[0].cuda())
                loss.backward()
                tot_loss += loss.item()
                nn.utils.clip_grad_norm_(self._model.parameters(), clipnorm)
                optimizer.step()

            print('='*100)
            print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))

        self._model.cpu()