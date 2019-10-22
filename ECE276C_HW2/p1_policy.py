"""
ECE 276C HW2 Question 1 PI and VI
"""
from copy import deepcopy
import numpy as np
# from p1_main import testPolicy


def policyEval(env, p_mat, r_mat, gamma, max_iter=50):
    """
    Policy iteration

    :param env: object, gym environment
    :param p_mat: np ndarray, transition probabilities matrix
    :param r_mat: np ndarray, reward function matrix
    :param gamma: float (0~1), discount factor
    :param max_iter: int, max iterations
    :return: a lambda policy function, take state as input and output action
             a list of success rate over episodes
    """
    assert isinstance(max_iter, int) and max_iter > 0
    assert 0 < gamma <= 1

    # get state space and action space dimension
    state_space_n = env.observation_space.n
    action_space_n = env.action_space.n

    # init placeholders
    V = np.zeros(state_space_n)
    Pi = np.zeros(state_space_n, dtype=int)
    success_rate = []

    for i in range(max_iter):
        print('Episode {}'.format(i), end='')
        # policy evaluation
        for state in range(state_space_n):
            v_old = V[state]

            action_s = Pi[state]
            sum_temp = 0
            for state_next in range(state_space_n):
                sum_temp += p_mat[state, action_s, state_next] * \
                    (r_mat[state, action_s, state_next] +
                        gamma * V[state_next])
            V[state] = sum_temp

        # policy improvement
        for j in range(100):
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
                print('    Pi stabilized in {} iterations'.format(j))
                break

        # eval
        success_rate.append(testPolicy(env, lambda state: Pi[state]))

    print('V^*: ', V)
    print('Pi^* ', Pi)
    return lambda state: Pi[state], success_rate


def valueIter(env, p_mat, r_mat, gamma, max_iter=50):
    """
    Value iteration

    :param env: object, gym environment
    :param p_mat: np ndarray, transition probabilities matrix
    :param r_mat: np ndarray, reward function matrix
    :param gamma: float (0~1), discount factor
    :param max_iter: int, max iterations
    :return: a lambda policy function, take state as input and output action
             a list of success rate over episodes
    """
    assert isinstance(max_iter, int) and max_iter > 0
    assert 0 < gamma <= 1

    # get state space and action space dimension
    state_space_n = env.observation_space.n
    action_space_n = env.action_space.n

    # init placeholders
    V = np.zeros(state_space_n)
    Pi = np.zeros(state_space_n, dtype=int)
    success_rate = []

    # TODO
    for i in range(max_iter):
        print('Episode {}'.format(i), end='')

        for j in range(100):
            delta = []
            for state in range(state_space_n):
                v_old = V[state]

                vs_temp = []
                for action in range(action_space_n):
                    sum_temp = 0
                    for state_next in range(state_space_n):
                        sum_temp += p_mat[state, action, state_next] * \
                            (r_mat[state, action, state_next] +
                             gamma * V[state_next])
                    vs_temp.append(sum_temp)
                V[state] = max(vs_temp)
                delta.append(abs(v_old - V[state]))

            if max(delta) < 1e-3:
                print('    V converged in {} iterations'.format(j))
                break

        # policy recovery
        for state in range(state_space_n):
            pis_temp = []
            for action in range(action_space_n):
                sum_temp = 0
                for state_next in range(state_space_n):
                    sum_temp += p_mat[state, action, state_next] * \
                        (r_mat[state, action, state_next] +
                            gamma * V[state_next])
                pis_temp.append(sum_temp)
            Pi[state] = np.argmax(pis_temp)

        # eval
        success_rate.append(testPolicy(env, lambda state: Pi[state]))

    print('V^*: ', V)
    print('Pi^* ', Pi)
    return lambda state: Pi[state], success_rate


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

    for _ in range(trials):
        state = env.reset()     # reset env
        done = False
        while not done:
            action = policy(state)
            state, _, done, _ = env.step(action)
        if state == 15:
            success += 1    # successfully reach goal

    success_rate = success / trials
    if verbose:
        print('success rate: {} ({} out of {})'.format(
            success_rate, success, trials))
    return success_rate
