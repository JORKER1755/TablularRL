# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import Env
from algo import QLearning
import time


def run_episode(env, agent, render=False, only_exploitation=False):
    total_steps = 0
    total_reward = 0
    decay = 1.0
    obs = env.reset()
    for _ in range(1000000):
        if only_exploitation:
            action = agent.predict(obs)
        else:
            action = agent.sample(obs)
        next_obs, reward, done = env.step(action)
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += decay*reward
        decay *= gamma
        total_steps += 1
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def tst_episode(env, agent):
    total_reward = 0
    decay = 1.0
    obs = env.reset()
    action_trajectory = ''
    for _ in range(60):
        action = agent.predict(obs)
        action_trajectory += env.ToChar[action]
        next_obs, reward, done = env.step(action)
        total_reward += decay*reward
        decay *= gamma
        obs = next_obs
        time.sleep(0.25)
        env.render()
        if done:
            break
    print('test reward = {0:.4f}'.format(total_reward))
    print('trajectory: {}'.format(action_trajectory))


def train():
    print('training...')
    env = Env(n=5, G=3)
    agent = QLearning(
        obs_n=env.n_obs,
        act_n=env.n_act,
        learning_rate=0.1,
        gamma=gamma,
        epsilon=0.1)
    is_render = False
    run_episode(env, agent, is_render, exploitation)
    agent.save(file_index)
    tst_episode(env, agent)


def tst():
    print('predict...')
    env = Env(n=5, G=3)
    agent = QLearning(
        obs_n=env.n_obs,
        act_n=env.n_act,
        learning_rate=0.1,
        gamma=gamma,
        epsilon=0.1)
    agent.restore(file_index)
    tst_episode(env, agent)


if __name__ == "__main__":
    file_index = 6
    gamma = 0.9
    exploitation = False

    # train()
    tst()
