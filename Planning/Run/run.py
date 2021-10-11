import gym
import numpy as np

from Planning.Algo.policy_iteration import PolicyIteration
from Planning.Algo.value_iteration import ValueIteration


def run_episode():
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    while True:
        env.render()
        obs, reward, done, _ = env.step(agent.predict(obs))
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate(n=100):
    scores = [run_episode() for _ in range(n)]
    # return np.mean(scores)
    print(np.mean(scores))


def train():
    while agent.learn():
        pass


if __name__ == '__main__':
    game_id = -1
    games = ['MountainCar-v0', 'CartPole-v0', 'Pong-ram-v0', 'Breakout-v0', 'Acrobot-v1', 'Pong-v0', 'FrozenLake-v0']
    env = gym.make(games[game_id])

    agent_id = 2
    if agent_id == 1:
        agent = PolicyIteration(env.env.nA, env.env.nS, env.env.P, gamma=1.)
    elif agent_id == 2:
        agent = ValueIteration(env.env.nA, env.env.nS, env.env.P, gamma=1.)

    evaluate(10)
    train()
    evaluate(10)
    env.close()
