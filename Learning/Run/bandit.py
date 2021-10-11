from TablularRL.Learning.Env.bandit import MultiArmedBandit
from TablularRL.Learning.Algo.bandit import BanditPreference


def train(n_steps):
    reward_sum = 0.0

    env.seed(0)
    for k in range(n_steps):
        action = algo.sample()
        reward = env.step(action)
        algo.learn(action, reward)

        reward_sum += reward
        if k % 10 == 0:
            print('current step: {}'.format(k))
            print('ideal values: {}'.format(env.reward_means))
            print('real values:  {}'.format(algo.probability))
            print('average reward: {}'.format(reward_sum/10))
            reward_sum = 0.0
            print('\n')


if __name__ == '__main__':
    env = MultiArmedBandit(10)
    algo = BanditPreference(env.actions, alpha=0.1)
    train(10000)
