"""
ECE 276C HW2 Question 1
"""
import gym
import numpy as np
from p1_policy import policyEval, valueIter


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
    counter_mat = np.zeros((state_space_n, action_space_n, state_space_n))

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
            counter_mat[state, action, state_next] += 1

            state = state_next
            counter += 1

    # normalize
    for i in range(trans_prob_mat.shape[0]):
        for j in range(trans_prob_mat.shape[1]):
            norm_temp = np.linalg.norm(trans_prob_mat[i,j,:], ord=1)
            if norm_temp != 0:
                trans_prob_mat[i,j,:] /= norm_temp

    counter_mat[counter_mat==0] = 1 # avoid singular
    reward_func_mat /= counter_mat
    return trans_prob_mat, reward_func_mat


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
    pi_policy = policyEval(env, p_mat, r_mat, gamma=0.99, max_iter=50)
    testPolicy(env, pi_policy)

    # Question 1.6
    print('\n===== Question 1.6 =====\n')
    print('Value Iteration')
    vi_policy = valueIter(env, p_mat, r_mat, gamma=0.99, max_iter=50)
    testPolicy(env, vi_policy)
