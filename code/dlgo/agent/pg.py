"""Policy gradient learning."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo.agent.base import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo import encoders
from dlgo import goboard_fast as goboard
from dlgo.rl.experience import ExperienceBuffer

__all__ = [
    'PolicyAgent',
    'load_policy_agent',
]


def normalize(x):
    total = np.sum(x)
    return x / total


class ExperienceDataSet(Dataset):
    def __init__(self, experience, transform=None):
        self.experience = experience
        self.transform = transform
    
    def __len__(self):
        return len(self.experience.states)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.experience.states, dtype=torch.float32)[idx]
        y = torch.tensor(self.experience.actions, dtype=torch.long)[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


class PolicyAgent(Agent):
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0
    
    def set_temperature(self, temperature):
        self._temperature = temperature
    
    def set_collector(self, collector):
        self._collector = collector

    def predict(self, input_tensor):
        return self._model(input_tensor.cuda()).cpu()

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height

        board_tensor = self._encoder.encode(game_state)
        input_tensor = torch.unsqueeze(torch.tensor(np.array(board_tensor), dtype=torch.float32), dim=0)

        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            move_probs = self.predict(input_tensor)

        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = torch.squeeze(move_probs, dim=0).detach().numpy()
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=board_tensor,
                        action=point_idx
                    )
                return goboard.Move.play(point)
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    def serialize(self, type='RL', version='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self._encoder.name(),
            'board_width': self._encoder.board_width,
            'board_height': self._encoder.board_height,
            'model_state_dict': self._model.state_dict(),
            'model': self._model,
        }, path + f"\\agents\\AlphaGo_Policy_{type}_Agent_{version}.pt")

    def train(self, winning_exp_buffer, losing_exp_buffer, lr=0.0001, clipnorm=1.0, batch_size=128):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        pivot = int(len(winning_exp_buffer.states) * 0.8)
        version='test'
        
        winning_train_buffer = ExperienceBuffer(
            winning_exp_buffer.states[:pivot],
            winning_exp_buffer.actions[:pivot],
            winning_exp_buffer.rewards[:pivot],
            []
        )
        losing_train_buffer = ExperienceBuffer(
            losing_exp_buffer.states[:pivot],
            losing_exp_buffer.actions[:pivot],
            losing_exp_buffer.rewards[:pivot],
            []
        )
        winning_test_buffer = ExperienceBuffer(
            winning_exp_buffer.states[pivot:],
            winning_exp_buffer.actions[pivot:],
            winning_exp_buffer.rewards[pivot:],
            []
        )
        losing_test_buffer = ExperienceBuffer(
            losing_exp_buffer.states[pivot:],
            losing_exp_buffer.actions[pivot:],
            losing_exp_buffer.rewards[pivot:],
            []
        )

        winning_train_dataset = ExperienceDataSet(winning_train_buffer)
        losing_train_dataset = ExperienceDataSet(losing_train_buffer)
        winning_test_dataset = ExperienceDataSet(winning_test_buffer)
        losing_test_dataset = ExperienceDataSet(losing_test_buffer)

        winning_train_loader = DataLoader(winning_train_dataset, batch_size=batch_size, shuffle=True)
        losing_train_loader = DataLoader(losing_train_dataset, batch_size=batch_size, shuffle=True)
        winning_test_loader = DataLoader(winning_test_dataset, batch_size=batch_size, shuffle=True)
        losing_test_loader = DataLoader(losing_test_dataset, batch_size=batch_size, shuffle=True)

        optimizer = SGD(self._model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        NUM_EPOCHES = 100
        self._model.cuda()
        total_steps = pivot // batch_size

        for epoch in range(NUM_EPOCHES):
            self._model.train()
            tot_loss = 0.0
            steps = 0

            for x, y in losing_train_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self._model(x)
                loss = 1 - loss_fn(y_, y.cuda())
                loss.backward()
                tot_loss += loss.item()
                # nn.utils.clip_grad_norm_(self._model.parameters(), clipnorm)
                optimizer.step()

                if steps >= total_steps // 2:
                    break
            
            for x, y in winning_train_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()
                y_ = self._model(x)
                loss = loss_fn(y_, y.cuda()) 
                loss.backward()
                tot_loss += loss.item()
                # nn.utils.clip_grad_norm_(self._model.parameters(), clipnorm)
                optimizer.step()

                if steps >= total_steps:
                    break

            print('='*50)
            print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))
            _, argmax = torch.max(y_, dim=1)
            train_acc = compute_acc(argmax, y)
            print("Epoch {}, Acc(train) : {}".format(epoch+1, train_acc))

            self._model.eval()
            eval_loss = 0
            x, y = next(iter(losing_test_loader))
            x = x.cuda()
            y_ = self._model(x)
            loss = loss_fn(y_, y.cuda()) 
            eval_loss += loss.item()
            x, y = next(iter(winning_test_loader))
            x = x.cuda()
            y_ = self._model(x)
            loss = loss_fn(y_, y.cuda()) 
            eval_loss += loss.item()
            eval_loss /= 2
            print("Epoch {}, Loss(val) : {}".format(epoch+1, eval_loss))
            _, argmax = torch.max(y_, dim=1)
            test_acc = compute_acc(argmax, y)
            print("Epoch {}, Acc(val) : {}".format(epoch+1, test_acc))

            torch.save({
                'model_state_dict': self._model.state_dict(),
                'loss': eval_loss,
            }, path + f"\\checkpoints\\alphago_rl_policy_epoch_{epoch+1}_v{version}.pt")

        self._model.cpu()


def load_policy_agent(type='SL', version='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\AlphaGo_Policy_{type}_Agent_{version}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height)
    )
    return PolicyAgent(model, encoder)

def compute_acc(argmax, y):
        count = 0
        for i in range(len(argmax)):
            if argmax[i] == y[i]:
                count += 1
        return count / len(argmax)