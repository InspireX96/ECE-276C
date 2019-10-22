"""
ECE 276C HW2 Question 2
"""
import multiprocessing
from itertools import repeat
import gym
import numpy as np
from sklearn.utils.extmath import cartesian
from p1_policy import testPolicy


class QlearningPolicy(object):
    """
    Train and test Q-learning policy
    """

    def __init__(self, env, alpha, gamma):
        """
        :param env: object, gym environment
        :param alpha: float, learning rate
        :param gamma: float (0~1), discount factor
        """
        assert 0 < gamma <= 1
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        # get state and action space
        self.state_space_n = env.observation_space.n
        self.action_space_n = env.action_space.n

        # init Q function
        self.Q_table = np.zeros((
            self.state_space_n, self.action_space_n))

    def _takeAction(self, state, episode):
        """
        Take action with epsilon-greedy (with probability 1 - episode/5000)

        :param state: int, current state
        :param episode: int, current episode
        :return: int, action
        """
        if np.random.rand() <= 1 - episode / 5000:
            # greedy
            return self.env.action_space.sample()
        # not greedy
        return np.argmax(self.Q_table[state, :])

    def updateQ(self, state, state_next, action, reward):
        """
        Q learning update

        :param state: int, current state
        :param state_next: int, next state
        :param action: int, action
        :param reward: float, reward
        """
        Q_current = self.Q_table[state, action]
        Q_next = self.Q_table[state_next, :]    # this is array

        # update Q function
        self.Q_table[state, action] += self.alpha * \
            (reward + self.gamma * np.max(Q_next) - Q_current)

    def trainPolicy(self, max_episode=5000, verbose=False):
        """
        Train policy

        :param max_episode: int (>0), training episode, defaults to 5000
        :param verbose: bool, flag to print more infomation and render final state
        :return: a lambda policy function, take state as input and output action
        """
        assert isinstance(max_episode, int) and max_episode > 0

        for i in range(max_episode):
            state = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = self._takeAction(state=state, episode=i)
                state_next, reward, done, _ = self.env.step(action)
                self.updateQ(state=state, state_next=state_next,
                             action=action, reward=reward)
                state = state_next
                steps += 1

        if verbose:
            print('Episode {} finished in {} steps'.format(i, steps))
        return lambda state: np.argmax(self.Q_table[state, :])


def build_search_grid(**kwargs):
    """
    Build search grid of alpha and gamma

    :param alpha_limit: array-like (len=2), lower and upper limit of alpha
    :param alpha_num: int, number of alpha samples
    :param gamma_list: array-like (len=2), lower and upper limit of gamma
    :param gamma_num: int, number of gamma samples
    :param use_random: bool, flag to use random sampling
    :return: M x N np ndarray, search grid, first column is alpha and second column is gamma
    """
    # parse kwargs
    alpha_limit, alpha_num = kwargs.get(
        'alpha_limit', []),  kwargs.get('alpha_num', None)
    if not alpha_limit:
        raise ValueError('alpha limit not defined!')
    if not alpha_num:
        raise ValueError('alpha num not defined!')
    gamma_limit, gamma_num = kwargs.get(
        'gamma_limit', []), kwargs.get('gamma_num', None)
    if not gamma_limit:
        raise ValueError('gamma limit not defined!')
    if not gamma_num:
        raise ValueError('gamma num not defind!')
    use_random = kwargs.get('random', False)
    print('Building grid with: {}'.format(kwargs))

    # random sample
    if use_random:
        print('Building grid using random sampling')
        alpha_list = min(alpha_limit) + (max(alpha_limit) -
                                         min(alpha_limit)) * np.random.rand(alpha_num)
        gamma_list = min(gamma_limit) + (max(gamma_limit) -
                                         min(gamma_limit)) * np.random.rand(gamma_num)

    else:
        print('Building grid using linspace')
        # linspace
        alpha_list = np.linspace(min(alpha_limit), max(alpha_limit), alpha_num)
        gamma_list = np.linspace(min(gamma_limit), max(gamma_limit), gamma_num)

    return cartesian((alpha_list, gamma_list))


def eval_policy(sample, env):
    """
    Eval policies using multiprocess

    :param sample: array-like (len=2), [alpha, gamma]
    :param env: object, gym environment
    :return: list, [alpha, gamma, success_rate]
    """
    alpha, gamma = sample[0], sample[1]
    qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
    policy = qlearn.trainPolicy()
    success_rate = testPolicy(env, policy, trials=100, verbose=False)
    print('alpha: {}, gamma: {}, success rate: {}'.format(alpha, gamma, success_rate))
    return [alpha, gamma, success_rate]


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    # Q 2.1(a)
    print('\n===== Question 2.1(a) =====\n')
    gamma = 0.99
    for alpha in [0.05, 0.1, 0.25, 0.5]:
        print('alpha: {}, gamma: {}'.format(alpha, gamma))
        qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
        policy = qlearn.trainPolicy()
        testPolicy(env, policy, trials=100, verbose=True)

    # Q 2.1(b)
    print('\n===== Question 2.1(b) =====\n')
    alpha = 0.05
    for gamma in [0.9, 0.95, 0.99]:
        print('alpha: {}, gamma: {}'.format(alpha, gamma))
        qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
        policy = qlearn.trainPolicy()
        testPolicy(env, policy, trials=100, verbose=True)

    # Q 2.2
    print('\n===== Question 2.2 =====\n')
    # grid search
    search_grid = build_search_grid(alpha_limit=[0.05, 0.5], alpha_num=10,
                                    gamma_limit=[0.8, 0.95], gamma_num=10,
                                    random=False)
    # grid_search_res = {}
    # grid_total_length = search_grid.shape[0]
    # for i in range(grid_total_length):
    #     [alpha, gamma] = search_grid[i]
    #     print('[{}/{}] Testing alpha: {}, gamma: {}'.format(i + 1,
    #                                                         grid_total_length, alpha, gamma))
    #     qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
    #     policy = qlearn.trainPolicy()
    #     success_rate = testPolicy(env, policy, trials=100, verbose=False)
    #     grid_search_res[(alpha, gamma)] = success_rate

    # sorted_grid_search_res = sorted(
    #     grid_search_res.items(), key=lambda x: x[1], reverse=True)
    # print('Best result: [alpha: {}, gamma: {}, success rate: {}]'.format(
    #     sorted_grid_search_res[0][0][0], sorted_grid_search_res[0][0][1], sorted_grid_search_res[0][1]))
    with multiprocessing.Pool() as pool:
        grid_search_res = pool.starmap(
            eval_policy, zip(search_grid, repeat(env)))
    sorted_grid_search_res = sorted(
        grid_search_res, key=lambda x: x[-1], reverse=True)
    print('Best result: [alpha: {}, gamma: {}, success rate: {}]'.format(
        sorted_grid_search_res[0][0], sorted_grid_search_res[0][1], sorted_grid_search_res[0][2]))

