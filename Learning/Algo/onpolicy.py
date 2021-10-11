import numpy as np
import pandas as pd

"""on-policy temporal-difference control"""


class Sarsa:
    """
    on-policy one-step temporal-difference control
    """
    def __init__(self, actions, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # index: state; columns: out
        self.values = pd.DataFrame(columns=self.actions, dtype=np.float)

    def save_new_state(self, state):
        to_be_append = pd.Series(
            np.zeros(self.actions.shape, dtype=np.float32),
            index=self.values.columns,
            name=state
        )
        self.values = self.values.append(to_be_append)  # 按行追加

    def sample(self, state):
        if state not in self.values.index:
            self.save_new_state(state)
            action = np.random.choice(self.actions)
            return action

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values.loc[state])]       # 没有处理多个相同最大值

    def learn(self, state, action, reward, state_, action_, done):
        # 可以取消对done的判断，因为self.values保存了终止状态的动作值，且值为零
        if done:
            target_value = reward
        else:
            target_value = reward + self.gamma*self.values.loc[state_, action_]
        td_error = target_value - self.values.loc[state, action]
        self.values.loc[state, action] += self.alpha*td_error


class ExpectedSarsa:
    """
    """
    def __init__(self, actions, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.stochastic_action_prob = self.epsilon/self.actions.shape[0]
        # index: state; columns: out
        self.values = pd.DataFrame(columns=self.actions, dtype=np.float)

    def save_new_state(self, state):
        to_be_append = pd.Series(
            np.zeros(self.actions.shape, dtype=np.float32),
            index=self.values.columns,
            name=state
        )
        self.values = self.values.append(to_be_append)  # 按行追加

    def sample(self, state):
        if state not in self.values.index:
            self.save_new_state(state)
            action = np.random.choice(self.actions)
            return action

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values.loc[state])]

    def learn(self, state, action, reward, state_, done):
        if done:
            target_value = reward
        else:
            value = self.values.loc[state_]
            expected_value = self.stochastic_action_prob*np.sum(value)+(1-self.epsilon)*np.argmax(value)
            target_value = reward + self.gamma*expected_value
        td_error = target_value - self.values.loc[state, action]
        self.values.loc[state, action] += self.alpha*td_error


class NStepSarsa:
    """
    """
    def __init__(self, actions, n_step, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.actions = actions
        self.n_step = n_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # index: state; columns: out
        self.values = pd.DataFrame(columns=self.actions, dtype=np.float)

        self.step_count = 0

        self.state_memory = np.empty(self.n_step)
        self.action_memory = np.empty(self.n_step)
        self.reward_memory = np.zeros(self.n_step, dtype=np.float)

    def reset(self):
        self.step_count = 0

    def save_new_state(self, state):
        to_be_append = pd.Series(
            np.zeros(self.actions.shape, dtype=np.float),
            index=self.values.columns,
            name=state
        )
        self.values = self.values.append(to_be_append)  # 按行追加

    def sample(self, state):
        if state not in self.values.index:
            self.save_new_state(state)
            action = np.random.choice(self.actions)
            return action

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values.loc[state])]

    def learn(self, state, action, reward, state_, action_, done):
        memory_index = self.step_count % self.n_step
        self.state_memory[memory_index] = state
        self.action_memory[memory_index] = action
        self.reward_memory[memory_index] = reward
        self.step_count += 1
        if self.step_count < self.n_step:   # 前n-1步不需要更新
            return

        if done:
            # backup方式计算target_value
            # 更新后n个动作值
            target_value = 0.0
            for i in range(1, self.n_step+1):
                memory_index = (self.step_count - i) % self.n_step
                target_value = self.gamma*target_value + self.reward_memory[memory_index]
                self.update(target_value, memory_index)
        else:
            # 更新往前第n步的动作值
            target_value = self.values.loc[state_, action_]
            for i in range(1, self.n_step+1):
                memory_index = (self.step_count - i) % self.n_step
                target_value = self.gamma*target_value + self.reward_memory[memory_index]
            self.update(target_value, memory_index)

    def update(self, target_value, memory_index):
        """更新动作值
        :param target_value: 目标动作值
        :param memory_index: 指定状态-动作对
        """
        state, action = self.state_memory[memory_index], self.action_memory[memory_index]
        td_error = target_value - self.values.loc[state, action]
        self.values.loc[state, action] += self.alpha * td_error
