import numpy as np
import pandas as pd


"""off-policy temporal-difference control"""


class QLearning:
    """
    off-policy one-step temporal-difference control
    """
    def __init__(self, actions, alpha=0.1, epsilon=0.1, gamma=0.9):
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
            action = self.predict(state)    # 相同最优值的动作如何处理？？？
            # out = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values.loc[state])]

    def learn(self, state, action, reward, state_, done):
        if done:
            target_value = reward
        else:
            target_value = reward + self.gamma*np.max(self.values.loc[state_])
        td_error = target_value - self.values.loc[state, action]
        self.values.loc[state, action] += self.alpha*td_error


class DoubleQLearning:
    """
    off-policy one-step temporal-difference control
    """
    def __init__(self, actions, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # index: state; columns: out
        self.values1 = pd.DataFrame(columns=self.actions, dtype=np.float)
        self.values2 = pd.DataFrame(columns=self.actions, dtype=np.float)

    def save_new_state(self, state):
        to_be_append = pd.Series(
            np.zeros(self.actions.shape, dtype=np.float32),
            index=self.actions,
            name=state
        )
        self.values1 = self.values1.append(to_be_append)  # 按行追加
        self.values2 = self.values2.append(to_be_append)  # 按行追加

    def sample(self, state):
        if state not in self.values1.index:
            self.save_new_state(state)
            action = np.random.choice(self.actions)
            return action

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.predict(state)    # 相同最优值的动作如何处理？？？
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values1.loc[state]+self.values2.loc[state])]

    def learn(self, state, action, reward, state_, done):
        if np.random.uniform() < 0.5:   # 选择更新哪个value_table
            evaluate_values, target_values = self.values1, self.values2
        else:
            evaluate_values, target_values = self.values2, self.values1

        if done:
            target_value = reward
        else:
            target_value = reward + self.gamma*target_values.loc[state_, np.argmax(evaluate_values.loc[state_])]
        td_error = target_value - evaluate_values.loc[state, action]
        evaluate_values.loc[state, action] += self.alpha*td_error


class NStepTreeBackup:
    """
    off-policy n-step temporal-difference control
    """
    def __init__(self, actions, n_step, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.actions = actions
        self.n_step = n_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # index: state; columns: out
        self.values = pd.DataFrame(columns=self.actions, dtype=np.float32)

        self.stochastic_action_prob = self.epsilon / self.actions.shape[0]

        self.step_count = 0

        self.state_memory = np.empty(self.n_step)
        self.action_memory = np.empty(self.n_step)
        self.value_memory = np.zeros(self.n_step, dtype=np.float32)
        self.td_error_memory = np.zeros(self.n_step, dtype=np.float32)
        self.policy_memory = np.zeros(self.n_step, dtype=np.float32)

    def reset(self):
        self.step_count = 0

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
            action = self.predict(state)    # 相同最优值的动作如何处理？？？
        return action

    def predict(self, state):
        return self.actions[np.argmax(self.values.loc[state])]

    def learn(self, state, action, reward, state_, action_, done):
        memory_index = self.step_count % self.n_step
        self.state_memory[memory_index] = state
        self.action_memory[memory_index] = action
        self.value_memory[memory_index] = self.values.loc[state, action]

        # 对于Sarsa类方法，值函数表保存了终止状态的动作值，且所有的值都保证是零，这里可以省略对done的判断
        value = self.values.loc[state_]     # 对终止状态也是有效的
        # expected out for epsilon-greedy policy
        expected_value = self.stochastic_action_prob * np.sum(value) + (1 - self.epsilon) * np.argmax(value)
        target_value = reward + self.gamma * expected_value
        td_error = target_value - self.value_memory[memory_index]
        self.td_error_memory[memory_index] = td_error
        # 以下实现方式不关心值函数是否保存终止状态情形，更加通用，但效率更低
        # if done:
        #     target_value = reward
        # else:
        #     out = self.values.loc[state_]
        #     # expected out for epsilon-greedy policy
        #     expected_value = self.stochastic_action_prob * np.sum(out) + (1 - self.epsilon) * np.argmax(out)
        #     target_value = reward + self.gamma * expected_value
        # td_error = target_value - self.value_memory[memory_index]
        # self.td_error_memory[memory_index] = td_error

        # policy_memory只需要n-1个值
        memory_index = (memory_index + 1) % self.n_step
        if value[action_] == np.argmax(value):  # greedy out
            policy_prob = 1 - self.gamma + self.stochastic_action_prob
        else:
            policy_prob = self.stochastic_action_prob
        self.policy_memory[memory_index] = policy_prob

        self.step_count += 1
        if self.step_count < self.n_step:   # 前n-1步不需要更新
            return

        if done:
            # backup方式计算target_value
            # 更新后n个动作值
            target_value = 0.0
            for i in range(1, self.n_step+1):
                memory_index = (self.step_count - i) % self.n_step
                target_value = target_value*self.gamma*self.policy_memory[memory_index] \
                    + self.td_error_memory[memory_index]
                self.update(target_value + self.value_memory[memory_index], memory_index)
        else:
            #
            target_value = 0.0
            for i in range(1, self.n_step+1):
                memory_index = (self.step_count - i) % self.n_step
                target_value = target_value*self.gamma*self.policy_memory[memory_index] \
                    + self.td_error_memory[memory_index]
            self.update(target_value + self.value_memory[memory_index], memory_index)

    def update(self, target_value, memory_index):
        """更新动作值
        :param target_value: 目标动作值
        :param memory_index: 指定状态-动作对
        """
        state, action = self.state_memory[memory_index], self.action_memory[memory_index]
        td_error = target_value - self.values.loc[state, action]
        self.values.loc[state, action] += self.alpha * td_error
