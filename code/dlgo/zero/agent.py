import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from dlgo.agent import Agent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.zero.experience import load_experience

__all__ = [
    'ZeroAgent',
    'load_zero_agent',
]


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {}
    
    def moves(self):
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        self.children[move] = child_node
    
    def has_child(self, move):
        return move in self.children
    
    def get_child(self, move):
        return self.children[move]
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count
    
    def prior(self, move):
        return self.branches[move].prior
    
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


class ZeroExperienceDataSet(Dataset):
    def __init__(self, experiencebuffers, transform=None):
        self.experiencebuffers = experiencebuffers
        self.transform = transform
        self.length = 0
        for buff in experiencebuffers:
            self.length += len(buff.advantages)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        div = idx // 1024
        mod = idx % 1024

        states = torch.tensor(self.experiencebuffers[div].states[mod], dtype=torch.float32)
        visit_counts = torch.tensor(self.experiencebuffers[div].visit_counts[mod], dtype=torch.float32)
        advantages = torch.tensor(self.experiencebuffers[div].advantages[mod], dtype=torch.float32)

        if self.transform:
            states = self.transform(states)

        return states, (visit_counts, advantages)


class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model.cuda()
        self.encoder = encoder
        self.collector = None
        self.num_rounds = rounds_per_move
        self.c = c
    
    def select_move(self, game_state):
        root = self.create_node(game_state)

        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                if node.state.is_over():
                    break
                next_move = self.select_branch(node)

            if not node.state.is_over():
                new_state = node.state.apply_move(next_move)
                child_node = self.create_node(new_state, move=next_move, parent=node)
            else:
                child_node = node
                node = node.parent
            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([root.visit_count(self.encoder.decode_move_index(idx)) 
                                     for idx in range(self.encoder.num_moves())])
            self.collector.record_decision(
                state=root_state_tensor, 
                visit_counts=visit_counts, 
                estimated_value=root.value
            )
        
        return max(root.moves(), key=root.visit_count)

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        
        return max(node.moves(), key=score_branch)
    
    def predict(self, input_tensor):
        output_tensor = self.model(input_tensor.cuda())
        return (output_tensor[0].cpu(), output_tensor[1].cpu())

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = torch.unsqueeze(torch.tensor(np.array(state_tensor), dtype=torch.float32), dim=0)
        priors, value = self.predict(model_input)
        priors = torch.squeeze(priors, dim=0).detach().numpy()
        value = torch.squeeze(value, dim=0).detach().numpy()
        # Add Dirichlet noise to the root node.
        if parent is None:
            noise = np.random.dirichlet(0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        move_priors = {
            self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
    
    def train(self, learning_rate=0.001, clipnorm=1.0, batch_size=128):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        version = 0
        base = path + f"\\buffers\\winning_experiences_zero_*.pt"
        files_num = len(glob.glob(base))
        pivot = int(files_num * 0.8)

        winning_train_buffers = []
        losing_train_buffers = []
        winning_test_buffers = []
        losing_test_buffers = []
        train_data_counts = 0

        for idx in range(files_num):
            if idx < pivot:
                experience_buffer = load_experience(result="winning", type="zero", no=f'{idx}')
                train_data_counts += len(experience_buffer.advantages)
                winning_train_buffers.append(experience_buffer)
                experience_buffer = load_experience(result="losing", type="zero", no=f'{idx}')
                train_data_counts += len(experience_buffer.advantages)
                losing_train_buffers.append(experience_buffer)
            else:
                experience_buffer = load_experience(result="winning", type="zero", no=f'{idx}')
                winning_test_buffers.append(experience_buffer)
                experience_buffer = load_experience(result="losing", type="zero", no=f'{idx}')
                losing_test_buffers.append(experience_buffer)

        winning_train_dataset = ZeroExperienceDataSet(winning_train_buffers, transform=trans_board)
        losing_train_dataset = ZeroExperienceDataSet(losing_train_buffers, transform=trans_board)
        winning_test_dataset = ZeroExperienceDataSet(winning_test_buffers, transform=trans_board)
        losing_test_dataset = ZeroExperienceDataSet(losing_test_buffers, transform=trans_board)

        winning_train_loader = DataLoader(winning_train_dataset, batch_size=batch_size, shuffle=True)
        losing_train_loader = DataLoader(losing_train_dataset, batch_size=batch_size, shuffle=True)
        winning_test_loader = DataLoader(winning_test_dataset, batch_size=batch_size, shuffle=True)
        losing_test_loader = DataLoader(losing_test_dataset, batch_size=batch_size, shuffle=True)

        optimizer = SGD(self.model.parameters(), lr=learning_rate)
        winning_loss_fn = CELoss
        losing_loss_fn = InverseCELoss
        policy_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()
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

                visit_sums = torch.sum(y[0], dim=1).reshape((y[0].shape[0], 1))
                y[0] = y[0] / visit_sums

                y_ = self.model(x)
                # loss = losing_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
                loss = policy_loss_fn((-1.0)*y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
                loss.backward()
                tot_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)
                optimizer.step()

                if steps >= total_steps // 2:
                    break
            
            for x, y in winning_train_loader:
                steps += 1
                optimizer.zero_grad()
                x = x.cuda()

                visit_sums = torch.sum(y[0], dim=1).reshape((y[0].shape[0], 1))
                y[0] = y[0] / visit_sums

                y_ = self.model(x)
                # loss = winning_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
                loss = policy_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
                loss.backward()
                tot_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)
                optimizer.step()

                if steps >= total_steps:
                    break

            print('='*50)
            print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))
            _, argmax_y_ = torch.max(y_[0], dim=1)
            _, argmax_y  = torch.max(y[0], dim=1)
            train_acc = compute_acc(argmax_y_, argmax_y)
            print("Epoch {}, Acc(train) : {}".format(epoch+1, train_acc))

            self.model.eval()
            eval_loss = 0
            x, y = next(iter(losing_test_loader))
            x = x.cuda()

            visit_sums = torch.sum(y[0], dim=1).reshape((y[0].shape[0], 1))
            y[0] = y[0] / visit_sums

            y_ = self.model(x)
            # loss = losing_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
            loss = policy_loss_fn((-1.0)*y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
            eval_loss += loss.item()
            x, y = next(iter(winning_test_loader))
            x = x.cuda()

            visit_sums = torch.sum(y[0], dim=1).reshape((y[0].shape[0], 1))
            y[0] = y[0] / visit_sums

            y_ = self.model(x)
            # loss = winning_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
            loss = policy_loss_fn(y_[0], y[0].cuda()) + value_loss_fn(y_[1], y[1].cuda())
            eval_loss += loss.item()
            eval_loss /= 2
            print("Epoch {}, Loss(val) : {}".format(epoch+1, eval_loss))
            _, argmax_y_ = torch.max(y_[0], dim=1)
            _, argmax_y  = torch.max(y[0], dim=1)
            test_acc = compute_acc(argmax_y_, argmax_y)
            print("Epoch {}, Acc(val) : {}".format(epoch+1, test_acc))

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'loss': eval_loss,
            }, path + f"\\checkpoints\\alphago_RL_zero_epoch_{epoch+1}_v{version}.pt")

        self.model.cpu()

    def serialize(self, version='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self.encoder.name(),
            'board_width': self.encoder.board_size,
            'board_height': self.encoder.board_size,
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
        }, path + f"\\agents\\AlphaGo_Zero_Agent_{version}.pt")


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

def CELoss(output, action):
    batch_size = len(output)
    feature_size = len(output[0])
    result = torch.zeros(batch_size)
    for i in range(batch_size):
        value = 0
        cum = torch.sum(torch.exp(output[i]))
        for j in range(feature_size):
            value += action[i][j] * torch.log(torch.exp(output[i][j]) / cum)
        result[i] = value
    return (-1.0) * torch.mean(result)

def InverseCELoss(output, action):
    batch_size = len(output)
    feature_size = len(output[0])
    result = torch.zeros(batch_size)
    for i in range(batch_size):
        value = 0
        cum = torch.sum(torch.exp(output[i]))
        for j in range(feature_size):
            value += action[i][j] * torch.log(1 - torch.exp(output[i][j]) / cum)
        result[i] = value
    return (-1.0) * torch.mean(result)

def load_zero_agent(version='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\AlphaGo_Zero_Agent_{version}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = get_encoder_by_name(
        encoder_name, (board_width, board_height)
    )
    return ZeroAgent(model, encoder)