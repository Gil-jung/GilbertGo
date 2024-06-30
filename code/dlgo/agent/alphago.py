import numpy as np
import torch
from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move


__all__ = [
    'AlphaGoNode',
    'AlphaGoMCTS'
]


class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent  # Tree nodes have one parent and potentially many children.
        self.children = {}

        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability  # A node is initialized with a prior probability.
        self.u_value = probability  # The utility function will be updated during search.
    
    def select_child(self):
        return max(
            self.children.items(),
            key=lambda child: child[1].q_value + child[1].u_value
        )
    
    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(parent=self, probability=prob)
    
    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)  # We update parents first to ensure we traverse the tree top to bottom.
        
        self.visit_count += 1  # Increment the visit count for this node.

        self.q_value += leaf_value / self.visit_count  # Add the specified leaf value to the Q-value, normalized by visit count.

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)  # <4> Update utility with current visit counts.


class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                lambda_value=0.5, num_simulations=10,
                depth=300, rollout_limit=10):
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent
        
        self.policy._model.cuda()
        self.rollout_policy._model.cuda()
        self.value.model.cuda()

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()
    
    def select_move(self, game_state):
        for simulation in range(self.num_simulations):  # From current state play out a number of simulations
            current_state = game_state
            node = self.root
            for depth in range(self.depth):  # Play moves until the specified depth is reached.
                if not node.children:  # If the current node doesn't have any children...
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)  # ... expand them with probabilities from the strong policy.
                    node.expand_children(moves, probabilities)
                
                if moves:
                    move, node = node.select_child()  # If there are children, we can select one and play the corresponding move.
                    current_state = current_state.apply_move(move)
                else:
                    current_state = current_state.apply_move(Move.pass_turn())

            current_state_tensor = self.value.encoder.encode(current_state)
            current_state_tensor = torch.unsqueeze(torch.tensor(np.array(current_state_tensor), dtype=torch.float32), dim=0)
            value_tensor = self.value.predict(current_state_tensor)  # Compute output of value network and a rollout by the fast policy.
            value = torch.squeeze(value_tensor, dim=1).detach().numpy()
            rollout = self.policy_rollout(current_state)

            weighted_value = (1 - self.lambda_value) * value + \
                self.lambda_value * rollout  # Determine the combined value function.
            
            node.update_values(weighted_value)  # Update values for this node in the backup phase
        
        move = max(self.root.children, key=lambda move:  # Pick most visited child of the root as next move.
                   self.root.children.get(move).visit_count)  
        
        self.root = AlphaGoNode()
        if move in self.root.children:  # If the picked move is a child, set new root to this child node.
            self.root = self.root.children[move]
            self.root.parent = None
        
        return move
    
    def policy_probabilities(self, game_state):
        encoder = self.policy._encoder
        
        game_state_tensor = encoder.encode(game_state)
        game_state_tensor = torch.unsqueeze(torch.tensor(np.array(game_state_tensor), dtype=torch.float32), dim=0)
        outputs_tensor = self.policy.predict(game_state_tensor)
        outputs = torch.squeeze(outputs_tensor, dim=0).detach().numpy()

        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            encoder = self.rollout_policy._encoder

            game_state_tensor = encoder.encode(game_state)
            game_state_tensor = torch.unsqueeze(torch.tensor(np.array(game_state_tensor), dtype=torch.float32), dim=0)
            move_probabilities_tensor = self.rollout_policy.predict(game_state_tensor)
            move_probabilities = torch.squeeze(move_probabilities_tensor, dim=0).detach().numpy()

            for idx in np.argsort(move_probabilities)[::-1]:
                max_point = encoder.decode_point_index(idx)
                greedy_move = Move(max_point)
                if greedy_move in game_state.legal_moves():
                    game_state = game_state.apply_move(greedy_move)
                    break

        next_player = game_state.next_player
        winner = game_state.winner()

        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0