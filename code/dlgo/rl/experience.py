import os
import torch
import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceCollector:
    def __init__(self):
        self.w_states = []
        self.w_actions = []
        self.w_rewards = []
        self.w_advantages = []
        self.l_states = []
        self.l_actions = []
        self.l_rewards = []
        self.l_advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
    
    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
    
    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        if reward > 0:
            self.w_states += self._current_episode_states
            self.w_actions += self._current_episode_actions
            self.w_rewards += [reward for _ in range(num_states)]

            for i in range(num_states):
                advantage = reward - self._current_episode_estimated_values[i]
                self.w_advantages.append(advantage)
        else:
            self.l_states += self._current_episode_states
            self.l_actions += self._current_episode_actions
            self.l_rewards += [reward for _ in range(num_states)]

            for i in range(num_states):
                advantage = reward - self._current_episode_estimated_values[i]
                self.l_advantages.append(advantage)
        
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, result='winning', name='v0'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'advantages': self.advantages,
        }, path + f"\\buffers\\{result}_experiences_{name}.pt")


def combine_experience(collectors):
    combined_w_states = np.concatenate([np.array(c.w_states) for c in collectors])
    combined_w_actions = np.concatenate([np.array(c.w_actions) for c in collectors])
    combined_w_rewards = np.concatenate([np.array(c.w_rewards) for c in collectors])
    combined_w_advantages = np.concatenate([np.array(c.w_advantages) for c in collectors])

    combined_l_states = np.concatenate([np.array(c.l_states) for c in collectors])
    combined_l_actions = np.concatenate([np.array(c.l_actions) for c in collectors])
    combined_l_rewards = np.concatenate([np.array(c.l_rewards) for c in collectors])
    combined_l_advantages = np.concatenate([np.array(c.l_advantages) for c in collectors])

    return ExperienceBuffer(
        combined_w_states,
        combined_w_actions,
        combined_w_rewards,
        combined_w_advantages,
    ), ExperienceBuffer(
        combined_l_states,
        combined_l_actions,
        combined_l_rewards,
        combined_l_advantages,
    )


def load_experience(name):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    winning_pt_file = torch.load(path + f"\\buffers\\winning_experiences_{name}.pt")
    losing_pt_file = torch.load(path + f"\\buffers\\losing_experiences_{name}.pt")
    
    return ExperienceBuffer(
        states=np.array(winning_pt_file['states']),
        actions=np.array(winning_pt_file['actions']),
        rewards=np.array(winning_pt_file['rewards']),
        advantages=np.array(winning_pt_file['advantages'])
    ), ExperienceBuffer(
        states=np.array(losing_pt_file['states']),
        actions=np.array(losing_pt_file['actions']),
        rewards=np.array(losing_pt_file['rewards']),
        advantages=np.array(losing_pt_file['advantages'])
    )