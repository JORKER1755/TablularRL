# -*- coding: utf-8 -*-

import numpy as np


class QLearning(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.random.normal(size=(obs_n, act_n))
        # print(self.Q)

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        action_list = np.where(Q_list == np.max(Q_list))[0]
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self, index=0):
        npy_file = './q_table{}.npy'.format(index)
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, index=0):
        npy_file = './q_table{}.npy'.format(index)
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
        # print(self.Q)
