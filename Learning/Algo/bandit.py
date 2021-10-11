import numpy as np
'''
单状态离散动作可变步长增量式算法   
'''


class BanditEpsilonGreedy:
    """动作价值+可变步长增量式+ε-greedy"""
    def __init__(self, actions, epsilon=0.1):
        self.actions = actions
        self.epsilon = epsilon
        self.values = np.zeros_like(self.actions, dtype=np.float)
        self.ns_learn = np.zeros_like(self.actions)

    def sample(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.predict()
        return action

    def predict(self):
        return self.actions[np.argmax(self.values)]

    def learn(self, action, reward):
        self.ns_learn[action] += 1
        self.values[action] += (reward - self.values[action])/self.ns_learn[action]


class BanditUCB:
    """动作价值+可变步长增量式+UCB(upper confidence bound)"""
    def __init__(self, actions, coef_c=2.0):
        self.actions = actions
        self.coef_c = coef_c
        self.values = np.zeros_like(self.actions, dtype=np.float)
        self.n_learn = 0
        self.ns_learn = np.zeros_like(self.actions)

    def sample(self):
        # 需要避免除零，优先选取次数为零的动作
        # upper confidence bound (UCB)
        action = self.actions[np.argmax(self.values + self.coef_c*np.sqrt(np.log(self.n_learn)/(self.ns_learn + 1e-5)))]
        return action

    def predict(self):
        return self.actions[np.argmax(self.values)]

    def learn(self, action, reward):
        self.n_learn += 1
        self.ns_learn[action] += 1
        self.values[action] += (reward - self.values[action])/self.ns_learn[action]


class BanditPreference:
    """动作偏好+固定步长增量式+softmax分布采样"""
    def __init__(self, actions, alpha=0.1):
        self.actions = actions
        self.alpha = alpha
        self.preferences = np.zeros_like(self.actions, dtype=np.float)
        self.reward_mean = 0.0      # 对所有reward
        self.n_learn = 0

    def sample(self):
        return np.random.choice(self.actions, p=self.softmax(self.preferences))

    def predict(self):
        return self.actions[np.argmax(self.preferences)]

    @property
    def probability(self):
        return self.softmax(self.preferences)

    @staticmethod
    def softmax(preference):
        exp_preference = np.exp(preference)
        return exp_preference / np.sum(exp_preference)

    def learn(self, action, reward):
        self.n_learn += 1
        self.reward_mean += (reward - self.reward_mean)/self.n_learn
        mask = self.actions == action
        self.preferences += self.alpha*(reward - self.reward_mean)*(mask - self.softmax(self.preferences))
