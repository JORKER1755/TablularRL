

class FixedAlgo:
    def __init__(self, actions):
        self.actions = actions
        self.action_always = self.actions.sample()

    def sample(self, state):
        return self.action_always

    def predict(self):
        ...

    def learn(self, action, reward):
        ...


class RandomAlgo:
    def __init__(self, actions):
        self.actions = actions

    def sample(self, state):
        return self.actions.sample()

    def predict(self):
        ...

    def learn(self, action, reward):
        ...

