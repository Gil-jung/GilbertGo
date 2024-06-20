import os
import numpy as np
import torch

from dlgo.agent.base import Agent
from dlgo.agent.helper_fast import is_point_an_eye
from dlgo import encoders
import dlgo.goboard_fast as goboard


__all__ = [
    'DeepLearningAgent',
    'load_prediction_agent',
]


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = torch.unsqueeze(torch.tensor(encoded_state, dtype=torch.float32), dim=0)
        return self.model(input_tensor)

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        move_probs = torch.squeeze(move_probs, dim=0).detach().numpy()
        move_probs = move_probs ** 3  # Increase the distance between the move likely and least likely moves.
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)  # Prevent move probs from getting stuck at 0 or 1
        move_probs = move_probs / np.sum(move_probs)  # Re-normalize to get another probability distribution.

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs  # Sample potential candidates
        )
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                not is_point_an_eye(game_state.board, point, game_state.next_player):  # Starting from the top, find a valid move that doesn't reduce eye-space.
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # If no legal and non-self-destructive moves are left, pass.
    
    def serialize(self, name='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self.encoder.name(),
            'board_width': self.encoder.board_width,
            'board_height': self.encoder.board_height,
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
        }, path + f"\\agents\\DL_Agent_{self.model.name()}_{self.encoder.name()}_{name}.pt")
    
def load_prediction_agent(model_name='large', encoder_name='sevenplane', name='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\DL_Agent_{model_name}_{encoder_name}_{name}.pt")
    model = pt_file['model']
    encoder_name = pt_file['encoder_name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = pt_file['board_width']
    board_height = pt_file['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height)
    )
    return DeepLearningAgent(model, encoder)