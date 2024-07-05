import os
import torch
import numpy as np

__all__ = [
    'ZeroExperienceCollector',
    'ZeroExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ZeroExperienceCollector:
    def __init__(self):
        self.w_states = []
        self.w_visit_counts = []
        self.w_rewards = []
        self.w_advantages = []
        self.l_states = []
        self.l_visit_counts = []
        self.l_rewards = []
        self.l_advantages = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []
        self._current_episode_estimated_values = []
    
    def record_decision(self, state, visit_counts, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)
        self._current_episode_estimated_values.append(estimated_value)
    
    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        if reward > 0:
            self.w_states += self._current_episode_states
            self.w_visit_counts += self._current_episode_visit_counts
            self.w_rewards += [reward for _ in range(num_states)]

            for i in range(num_states):
                advantage = reward - self._current_episode_estimated_values[i]
                self.w_advantages.append(advantage)
        else:
            self.l_states += self._current_episode_states
            self.l_visit_counts += self._current_episode_visit_counts
            self.l_rewards += [reward for _ in range(num_states)]

            for i in range(num_states):
                advantage = reward - self._current_episode_estimated_values[i]
                self.l_advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_visit_counts = []
        self._current_episode_estimated_values = []


class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards, advantages):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, result='winning', name='zero'):
        path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
        torch.save({
            'states': self.states,
            'visit_counts': self.visit_counts,
            'rewards': self.rewards,
            'advantages': self.advantages,
        }, path + f"\\buffers\\{result}_experiences_{name}.pt")


def combine_experience(collectors):
    combined_w_states = np.concatenate([np.array(c.w_states) for c in collectors])
    combined_w_visit_counts = np.concatenate([np.array(c.w_visit_counts) for c in collectors])
    combined_w_rewards = np.concatenate([np.array(c.w_rewards) for c in collectors])
    combined_w_advantages = np.concatenate([np.array(c.w_advantages) for c in collectors])

    combined_l_states = np.concatenate([np.array(c.l_states) for c in collectors])
    combined_l_visit_counts = np.concatenate([np.array(c.l_visit_counts) for c in collectors])
    combined_l_rewards = np.concatenate([np.array(c.l_rewards) for c in collectors])
    combined_l_advantages = np.concatenate([np.array(c.l_advantages) for c in collectors])

    return ZeroExperienceBuffer(
        combined_w_states,
        combined_w_visit_counts,
        combined_w_rewards,
        combined_w_advantages,
    ), ZeroExperienceBuffer(
        combined_l_states,
        combined_l_visit_counts,
        combined_l_rewards,
        combined_l_advantages,
    )


def load_experience(result="winning", type="zero", no=0):
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    pt_file = torch.load(path + f"\\buffers\\{result}_experiences_{type}_{no}.pt")
    
    return ZeroExperienceBuffer(
        states=np.array(pt_file['states']),
        visit_counts=np.array(pt_file['visit_counts']),
        rewards=np.array(pt_file['rewards']),
        advantages=np.array(pt_file['advantages'])
    )