"""
ECE 276C HW2 Question 1
"""

import gym
import numpy as np


def toyPolicy(state):
    """
    Toy policy in problem 3

    :param state: current state
    :return: int, policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3)
    """
    return (state + 1) % 4


def testPolicy(env, policy, trials=100, verbose=False):
    """
    Test policy, return averate rate of successful episodes
    over 100 trials.

    :param env: object, gym environment
    :param policy: function, a deterministic policy that takes state as input,
                             output action in (0, 1, 2, 3, 4)
    :param trials: int (>0), number of trials to test policy, defaults to 100
    :param verbose: bool, flag to print more infomation and render final state
    """
    assert isinstance(trials, int) and trials > 0
    success = 0

    for i in range(trials):
        state = env.reset()     # reset env
        done = False
        while not done:
            action = policy(state)
            state, _, done, _ = env.step(action)
        if state == 15:
            success += 1    # successfully reach goal
        if verbose:
            print('\n**trial {} final state: {}**'.format(i, state))
            env.render()

    if verbose:
        print('success rate: {} ({} out of {})'.format(
            success / trials, success, trials))
    else:
        print('success rate: {}'.format(success / trials))


def learnModel(samples=5):
    """
    Learn model

    :param samples: int, random samples
    """
    assert isinstance(samples, int) and samples > 0
    state = env.reset()
    action = env.action_space.sample()   # random action
    state_next, reward, done, _ = env.step(action)
    print(state, state_next, reward)
    # TODO: wait for piazza clarification


def policyEval():
    """
    Policy iteration
    """
    pass


def valueIter():
    """
    Value iteration
    """
    pass


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    # Question 1.3
    print('\n===== Question 1.3 =====\n')
    testPolicy(env, toyPolicy, trials=100, verbose=False)
    # state, reward, done, info = env.step()
