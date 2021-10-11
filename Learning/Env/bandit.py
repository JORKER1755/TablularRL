import numpy as np


class MultiArmedBandit:

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.actions = np.arange(self.n_actions)
        self.reward_means = np.random.normal(0.0, 2.0, (self.n_actions, ))
        self.reward_variance = 1.0

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def reset(self):
        ...

    def step(self, action):
        reward = np.random.normal(self.reward_means[action], self.reward_variance)
        return reward

    def render(self):
        ...

    def close(self):
        ...
