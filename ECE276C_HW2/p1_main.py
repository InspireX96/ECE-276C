"""
ECE 276C HW2 Question 1
"""
import gym
import numpy as np
from matplotlib import pyplot as plt
from p1_policy import policyEval, valueIter, testPolicy


def toyPolicy(state):
    """
    Toy policy in problem 3

    :param state: current state
    :return: int, policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3)
    """
    return (state + 1) % 4

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
    testPolicy(env, toyPolicy, verbose=True)

    # Question 1.4
    p_mat, r_mat = learnModel()
    print(p_mat)
    print(r_mat)

    # Question 1.5
    print('\n===== Question 1.5 =====\n')
    print('Policy Iteration')
    pi_policy, pi_success_rate = policyEval(env, p_mat, r_mat, gamma=0.99, max_iter=50)
    testPolicy(env, pi_policy, verbose=True)
    # plot
    plt.figure()
    plt.plot(pi_success_rate)
    plt.title('Question 1.5 Policy Iteration')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.show()

    # Question 1.6
    print('\n===== Question 1.6 =====\n')
    print('Value Iteration')
    vi_policy, vi_success_rate = valueIter(env, p_mat, r_mat, gamma=0.99, max_iter=50)
    testPolicy(env, vi_policy, verbose=True)
    # plot
    plt.figure()
    plt.plot(pi_success_rate)
    plt.title('Question 1.6 Value Iteration')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.show()

    # TODO: savefig
