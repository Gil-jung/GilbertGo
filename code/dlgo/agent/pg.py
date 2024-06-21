"""Policy gradient learning."""
import os
import numpy as np
import torch

from dlgo.agent.base import Agent
from dlgo.agent.helper_fast import is_point_an_eye
from dlgo import encoders
from dlgo import goboard_fast as goboard

__all__ = [
    'PolicyAgent',
    'load_policy_agent',
    'policy_gradient_loss',
]


class PolicyAgent(Agent):
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0

    def predict(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        input_tensor = torch.unsqueeze(torch.tensor(encoded_state, dtype=torch.float32), dim=0)
        return self._model(input_tensor)

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height

        board_tensor = self._encoder.encode(game_state)
        x = np.array([board_tensor])

        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            move_probs = self._model.predict(x)

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

    def serialize(self, name='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'encoder_name': self._encoder.name(),
            'board_width': self._encoder.board_width,
            'board_height': self._encoder.board_height,
            'model_state_dict': self._model.state_dict(),
            'model': self._model,
        }, path + f"\\agents\\PG_Agent_{self._model.name()}_{self._encoder.name()}_{name}.pt")


def load_policy_agent(model_name='large', encoder_name='simple', name='v0'):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\agents\\PG_Agent_{model_name}_{encoder_name}_{name}.pt")
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