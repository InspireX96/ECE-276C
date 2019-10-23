"""
ECE 276C HW2 Question 2 grid search
"""
import multiprocessing
from itertools import repeat
import gym
import numpy as np
from sklearn.utils.extmath import cartesian
from p1_policy import testPolicy
from p2_main import QlearningPolicy


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


def eval_policy_on_grid(sample, env):
    """
    Eval policies

    :param sample: array-like (len=2), [alpha, gamma]
    :param env: object, gym environment
    :return: alpha, gamma, success_rate_test
    """
    env.reset()
    alpha, gamma = sample[0], sample[1]
    qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
    policy = qlearn.trainPolicy(disable_success_rate=True)
    success_rate_test = testPolicy(policy, trials=100, verbose=False)
    print('alpha: {}, gamma: {}, success rate: {}'.format(
        alpha, gamma, success_rate_test))
    return alpha, gamma, success_rate_test


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    # grid search
    search_grid = build_search_grid(alpha_limit=[0.05, 0.5], alpha_num=10,
                                    gamma_limit=[0.9, 1], gamma_num=11,
                                    random=False)

    with multiprocessing.Pool() as pool:
        grid_search_res = pool.starmap(
            eval_policy_on_grid, zip(search_grid, repeat(env)))
    sorted_grid_search_res = sorted(
        grid_search_res, key=lambda x: x[-1], reverse=True)
    alpha_best, gamma_best = sorted_grid_search_res[0][0], sorted_grid_search_res[0][1]
    print('Best result: [alpha: {}, gamma: {}, success rate: {}]'.format(
        alpha_best, gamma_best, sorted_grid_search_res[0][-1]))
