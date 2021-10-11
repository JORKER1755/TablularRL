import numpy as np


class ValueIteration:
    def __init__(self, n_action, n_state, model, gamma=0.9):
        self.n_action = n_action
        self.n_state = n_state
        self.model = model  # (probability, next state, reward, done)
        self.values = np.zeros(self.n_state)
        self.gamma = gamma
        self.eps = 1e-10    # 策略评估收敛判定条件

    def predict(self, state):
        qs = np.zeros(self.n_action)  # 当前状态下所有动作值
        for a in range(self.n_action):
            qs[a] = sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[state][a]])
        return np.argmax(qs)

    def learn(self):
        total_improve_value = 0.0
        for s in range(self.n_state):
            qs = np.zeros(self.n_action)  # 当前状态下所有动作值
            for a in range(self.n_action):
                qs[a] = sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]])
            q = np.max(qs)
            total_improve_value += np.fabs(q - self.values[s])
            self.values[s] = q
        return total_improve_value > self.eps       # 收敛返回False
