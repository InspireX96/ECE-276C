"""
ECE 276C HW2 Question 1
"""

from copy import deepcopy
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
    :return: float, success rate
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

    success_rate = success / trials
    if verbose:
        print('success rate: {} ({} out of {})'.format(
            success_rate, success, trials))
    else:
        print('success rate: {}'.format(success_rate))
    return success_rate


def learnModel(samples=int(1e5)):
    """
    Learn model, get transition probabilities and reward function

    :param samples: int, random samples, defaults to 10^5
    :returns: trans_prob_mat: np ndarray, transition probabilites[state, action, state_next]
              reward_func_mat: np ndarray, transition probabilites[state, action, state_next]
    """
    assert isinstance(samples, int) and samples > 0

    # get state space and action space dimension
    state_space_n = env.observation_space.n
    action_space_n = env.action_space.n

    # placeholders for transition prob and reward func
    # state, action, state_next
    trans_prob_mat = np.zeros((state_space_n, action_space_n, state_space_n))
    # state, action ,state_next
    reward_func_mat = np.zeros((state_space_n, action_space_n, state_space_n))

    counter = 0
    while counter < samples:
        state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()   # random action
            state_next, reward, done, _ = env.step(action)

            # update
            trans_prob_mat[state, action, state_next] += 1
            reward_func_mat[state, action, state_next] += reward

            state = state_next
            counter += 1

    # normalize
    trans_prob_mat /= samples
    reward_func_mat /= samples
    return trans_prob_mat, reward_func_mat


def policyEval(p_mat, r_mat, gamma, max_iter=50):
    """
    Policy iteration

    :param p_mat: np ndarray, transition probabilities matrix
    :param r_mat: np ndarray, reward function matrix
    :param gamma: float (0~1), discount factor
    :param max_iter: int, max iterations
    :return: a lambda policy function, take state as input and output action
    """
    assert isinstance(max_iter, int) and max_iter > 0
    assert 0 < gamma <= 1

    # get state space and action space dimension
    state_space_n = env.observation_space.n
    action_space_n = env.action_space.n

    # init placeholders
    V = np.zeros(state_space_n)
    Pi = np.zeros(state_space_n, dtype=int)

    for i in range(max_iter):
        # policy evaluation
        for state in range(state_space_n):
            action_s = Pi[state]

            sum_temp = 0
            # import ipdb; ipdb.set_trace()
            for state_next in range(state_space_n):
                sum_temp += p_mat[state, action_s, state_next] * \
                    (r_mat[state, action_s, state_next] + gamma * V[state_next])
            V[state] = sum_temp

        # policy improvement
        Pi_old = deepcopy(Pi)
        for state in range(state_space_n):
            V_temp = np.zeros(action_space_n)
            for action in range(action_space_n):
                sum_temp = 0
                for state_next in range(state_space_n):
                    sum_temp += p_mat[state, action, state_next] * \
                        (r_mat[state, action, state_next] +
                         gamma * V[state_next])
                V_temp[action] = sum_temp
            Pi[state] = np.argmax(V_temp)

        if (Pi == Pi_old).all():
            print('PI converged in iteration {}'.format(i))
    print('V^*: ', V)
    print('Pi^* ', Pi)
    return lambda state: Pi[state]


def valueIter(p_mat, r_mat, gamma, max_iter=50):
    """
    Value iteration

    :param p_mat: np ndarray, transition probabilities matrix
    :param r_mat: np ndarray, reward function matrix
    :param gamma: float (0~1), discount factor
    :param max_iter: int, max iterations
    :return: a lambda policy function, take state as input and output action
    """
    assert isinstance(max_iter, int) and max_iter > 0
    assert 0 < gamma <= 1

    # get state space and action space dimension
    state_space_n = env.observation_space.n
    action_space_n = env.action_space.n

    # init placeholders
    V = np.zeros(state_space_n)
    Pi = np.zeros(state_space_n, dtype=int)

    # TODO


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    # Question 1.3
    print('\n===== Question 1.3 =====\n')
    print('toy policy')
    testPolicy(env, toyPolicy, trials=100, verbose=False)

    # Question 1.4
    p_mat, r_mat = learnModel()
    print(p_mat)
    print(r_mat)

    # Question 1.5
    print('\n===== Question 1.5 =====\n')
    print('Policy Iteration')
    pi_policy = policyEval(p_mat, r_mat, gamma=0.99)
    testPolicy(env, pi_policy)

    # Question 1.6
