import numpy as np


class PolicyIteration:
    def __init__(self, n_action, n_state, model, gamma=0.9):
        self.n_action = n_action
        self.n_state = n_state
        self.model = model  # 转移概率等
        self.policy = np.random.choice(self.n_action, size=self.n_state)  # initialize a random policy
        self.values = np.zeros(self.n_state)
        self.gamma = gamma
        self.eps = 1e-10    # 策略评估收敛判定条件

    def predict(self, state):
        return self.policy[state]

    def evaluate_policy(self):
        while True:
            total_improve_value = 0.0
            for s in range(self.n_state):
                a = self.policy[s]
                q = sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]])
                total_improve_value += np.fabs(q - self.values[s])
                self.values[s] = q
            if total_improve_value < self.eps:  # 收敛
                break

    def improve_policy(self):
        n_changed_action = 0
        for s in range(self.n_state):
            qs = np.zeros(self.n_action)    # 当前状态下所有动作值，由于使用的是状态价值函数而不是动作价值函数，因此需要额外计算量
            for a in range(self.n_action):
                qs[a] = sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]])
            new_a = np.argmax(qs)
            n_changed_action += self.policy[s] != new_a
            self.policy[s] = new_a
        return n_changed_action > 0

    def learn(self):
        self.evaluate_policy()
        improved = self.improve_policy()
        return improved     # done
