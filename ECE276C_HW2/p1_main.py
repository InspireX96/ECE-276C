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


def testPolicy(policy):
    """
    Test policy, return averate rate of successful episodes
    over 100 trials.

    Note: use global variable env 

    :param policy: function, a deterministic policy that takes state as input,
                             output action in (0, 1, 2, 3, 4)
    """
    trials = 100    # 100 trials
    success = np.zeros(trials)  # array to record success/fail

    for i in range(trials):
        state = env.reset()     # reset env
        done = False
        while not done:
            action = policy(state)
            state, reward, done, info = env.step(action)
        if state == 15:
            success[i] = 1

    print('success rate: ', np.sum(success) / trials)


def learnModel(samples=5):
    """
    Learn model
    """
    assert isinstance(samples, int) and samples > 0
    state = env.reset()
    action = np.random.randint(4)   # random action
    state_next, reward, done, info = env.step(action)
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
    env.render()

    # Question 1.3
    testPolicy(toyPolicy)
    # state, reward, done, info = env.step()
