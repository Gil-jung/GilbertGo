import os
import torch
import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceBuffer(object):
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, name='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'advantages': self.advantages,
        }, path + f"\\buffers\\experience_{name}.pt")


def load_experience(name):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\buffers\\experience_{name}.pt")
    
    return ExperienceBuffer(
        states=np.array(pt_file['states']),
        actions=np.array(pt_file['actions']),
        rewards=np.array(pt_file['rewards']),
        advantages=np.array(pt_file['advantages'])
    )