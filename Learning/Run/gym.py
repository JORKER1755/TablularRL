"""
MountainCar-v0
Pong-v0             乒乓球
Pong-ram-v0         乒乓球
Acrobot-v1          两截倒立摆
Breakout-v0         打砖块
CartPole-v0         倒立摆
FrozenLake-v0       冰湖      冰面上路径规划，掉入hole会死掉
FrozenLake8x8-v0
Montezuma’s Revenge
Private Eye
CliffWalking-v0
MsPacman-v0
Hopper-v1
Go9x9-v0			围棋

class Env:
    env.action_space
    env.observation_space
    env.reward_range
    env.metadata				字典方式保存render模式等信息

gym.envs：
    envs.register()
    注册自定义环境，供gym.make()加载
    envs.registry.all()
    查看所有环境

gym.spaces：
    spaces.Discrete(8)

gym.wrappers：
    env = wrappers.Monitor(env, root_dir=outdir, force=True)

gym.logger：

数值：
    float('inf')	正无穷
    -float('inf')	负无穷

数据类型：
    Discrete		n个离散状态，取值为0,1,...,n-1
    Box				n维连续状态，每一维取值均设置有上下限

"""

import gym

from TablularRL.Learning.Algo.simple import RandomAlgo, FixedAlgo
from TablularRL.Learning.Algo.control import MountainCarAlgo


def predict(episodes=10, max_step=100):
    for i_episode in range(episodes):
        observation = env.reset()
        print(observation)
        for t in range(max_step):
            env.render()
            action = agent.sample(observation)
            observation, reward, done, info = env.step(action)
            print('episode {}-step {}, taking out {}, observation {}'.format(i_episode, t, action, observation))
            if done:
                print("Episode finished after {} time steps".format(t + 1))
                break


# import gym
# env = gym.make('MountainCar-v0')
# print('观测空间 = {}'.format(env.observation_space))
# print('动作空间 = {}'.format(env.action_space))
# print('观测范围 = {} ~ {}'.format(env.observation_space.low,
#         env.observation_space.high))
# print('动作数 = {}'.format(env.action_space.n))


if __name__ == '__main__':
    game_id = -1
    games = ['MountainCar-v0', 'CartPole-v0', 'Pong-ram-v0', 'Breakout-v0', 'Acrobot-v1', 'Pong-v0', 'FrozenLake-v0']
    env = gym.make(games[game_id])

    agent_id = 1
    if agent_id == 1:
        agent = RandomAlgo(env.action_space)
    elif agent_id == 2:
        agent = FixedAlgo(env.action_space)
    else:
        agent = MountainCarAlgo(env.action_space)

    predict()

    env.close()
