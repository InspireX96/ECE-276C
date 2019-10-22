"""
ECE 276C HW2 Question 1 PI and VI
"""
from copy import deepcopy
import numpy as np


def policyEval(env, p_mat, r_mat, gamma, max_iter=50):
    """
    Policy iteration

    :param env: object, gym environment
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

    print('V^*: ', V)
    print('Pi^* ', Pi)
    return lambda state: Pi[state]


def valueIter(env, p_mat, r_mat, gamma, max_iter=50):
    """
    Value iteration

    :param env: object, gym environment
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
                # import ipdb; ipdb.set_trace()
                print('    V converged in {} iterations'.format(j))
                break

    # TODO: policy recovery
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

    print('V^*: ', V)
    print('Pi^* ', Pi)
    return lambda state: Pi[state]
