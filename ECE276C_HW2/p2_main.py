"""
ECE 276C HW2 Question 2
"""
import time
import gym
import numpy as np
from matplotlib import pyplot as plt
from p1_policy import testPolicy


class QlearningPolicy(object):
    """
    Train and test Q-learning policy
    """

    def __init__(self, env, alpha, gamma, sigma=0.6, delta=0.25):
        """
        :param env: object, gym environment
        :param alpha: float, learning rate
        :param gamma: float (0~1), discount factor
        :param sigma: float, for adaptive epsilon greedy only, inverse sensitivity (0~1)
        :param delta: float, for adaptive epsilon greedy only, influence parameter
        """
        assert 0 < gamma <= 1
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        self.sigma = sigma
        self.delta = delta

        # get state and action space
        self.state_space_n = env.observation_space.n
        self.action_space_n = env.action_space.n

        # init Q function
        # self.Q_table = np.zeros((
        #     self.state_space_n, self.action_space_n))
        self.Q_table = np.random.random((
            self.state_space_n, self.action_space_n))

        # record
        self.success_rate = []
        self.eps_list = []

    def _takeAction(self, state, episode):
        """
        Take action with epsilon-greedy (with probability 1 - episode/5000)

        :param state: int, current state
        :param episode: int, current episode
        :return: int, action
        """
        epsilon = 1 - episode / 5000
        self.eps_list.append(epsilon)
        if np.random.rand() <= epsilon:
            # greedy
            return self.env.action_space.sample()
        # not greedy
        return np.argmax(self.Q_table[state, :])

    def _takeActionAdaptive(self, state, epsilon):
        """
        Take action with adaptive epsilon-greedy

        :param state: int, current state
        :param epsilon: float, epsilon
        :return: int, action
        """
        self.eps_list.append(epsilon)
        if np.random.rand() <= epsilon:
            # greedy
            return self.env.action_space.sample()
        # not greedy
        return np.argmax(self.Q_table[state, :])

    def _calculateEpsilonVDBE(self, epsilon, state, state_next, action):
        """
        Calculate epsilon value using VDBE, from paper:
        https://link.springer.com/content/pdf/10.1007%2F978-3-642-16111-7_23.pdf

        :param epsilon: float, epsilon
        :param state: int, state
        :param state_next: int, next state
        :param action: int, action
        :return: float, next epsilon
        """
        Q_current = self.Q_table[state, action]
        Q_next = self.Q_table[state_next, action]
        f = (1 - np.exp(-abs(Q_next - Q_current) / self.sigma)) / \
            (1 + np.exp(-abs(Q_next - Q_current) / self.sigma))
        return self.delta * f + (1 - self.delta) * epsilon

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

    def trainPolicy(self, max_episode=5000, verbose=False, disable_success_rate=False, adaptive_greedy=False):
        """
        Train policy

        :param max_episode: int (>0), training episode, defaults to 5000
        :param verbose: bool, flag to print more infomation and render final state
        :param disable_success_rate: bool, flag to disable recording success rate
        :param adaptive_greedy: bool, flag to turn on adaptive epsilon greedy
        :returns: a lambda policy function, take state as input and output action
        """
        assert isinstance(max_episode, int) and max_episode > 0
        if adaptive_greedy:
            eps_adaptive = 1

        for i in range(max_episode):
            state = self.env.reset()
            done = False
            steps = 0
            while not done:
                # take action
                if adaptive_greedy:
                    action = self._takeActionAdaptive(
                        state=state, epsilon=eps_adaptive)
                else:
                    action = self._takeAction(state=state, episode=i)
                # step
                state_next, reward, done, _ = self.env.step(action)
                # update
                if adaptive_greedy:
                    eps_adaptive = self._calculateEpsilonVDBE(
                        epsilon=eps_adaptive, state=state, state_next=state_next, action=action)
                self.updateQ(state=state, state_next=state_next,
                             action=action, reward=reward)
                state = state_next
                steps += 1

            # eval policy
            if not disable_success_rate:
                if i % 100 == 0:
                    self.success_rate.append(testPolicy(
                        lambda state: np.argmax(self.Q_table[state, :])))

            if verbose:
                print('Episode {} finished in {} steps'.format(i, steps))

        return lambda state: np.argmax(self.Q_table[state, :])


def eval_policy(sample, env, adaptive_greedy=False):
    """
    Eval policies

    :param sample: array-like (len=2), [alpha, gamma]
    :param env: object, gym environment
    :param adaptive_greedy: bool, flag to turn on adaptive epsilon greedy
    :return: success_rate_train(list), eps_list(list), success_rate_test
    """
    env.reset()
    alpha, gamma = sample[0], sample[1]
    qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
    policy = qlearn.trainPolicy(
        disable_success_rate=False, adaptive_greedy=adaptive_greedy)
    success_rate_train = qlearn.success_rate
    success_rate_test = testPolicy(policy, trials=100, verbose=False)
    eps_list = qlearn.eps_list
    print('alpha: {}, gamma: {}, success rate: {}'.format(
        alpha, gamma, success_rate_test))
    return success_rate_train, eps_list, success_rate_test


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()

    # Q 2.1(a)
    print('\n===== Question 2.1(a) =====\n')
    gamma = 0.99
    plt.figure(figsize=(20, 10))
    for i, alpha in enumerate([0.05, 0.1, 0.25, 0.5]):
        [success_rate_train, _, _] = eval_policy((alpha, gamma), env)
        time.sleep(0.1)
        # plot
        plt.subplot(2, 2, i+1)
        plt.plot(success_rate_train)
        plt.title(
            'Q Learning Success Rate (alpha = {}, gamma = {})'.format(alpha, gamma))
        plt.xlabel('Episode * 100')
        plt.ylabel('Success rate')
    plt.savefig('Question 21a.png')
    plt.draw()
    plt.waitforbuttonpress(timeout=5)
    plt.close()

    # Q 2.1(b)
    print('\n===== Question 2.1(b) =====\n')
    alpha = 0.05
    plt.figure(figsize=(20, 10))
    for i, gamma in enumerate([0.9, 0.95, 0.99]):
        # print('alpha: {}, gamma: {}'.format(alpha, gamma))
        [success_rate_train, _, _] = eval_policy((alpha, gamma), env)
        time.sleep(0.1)
        # plot
        plt.subplot(2, 2, i+1)
        plt.plot(success_rate_train)
        plt.title(
            'Q Learning Success Rate (alpha = {}, gamma = {})'.format(alpha, gamma))
        plt.xlabel('Episode * 100')
        plt.ylabel('Success rate')
    plt.savefig('Question 21b.png')
    plt.draw()
    plt.waitforbuttonpress(timeout=5)
    plt.close()

    # Q 2.2
    print('\n===== Question 2.2 =====\n')
    # best parameters from p2_gridsearch.py
    alpha = 0.1
    gamma = 0.99

    # compare
    # without adaptive epsilon greedy
    plt.figure(figsize=(20, 10))
    print('Not using adaptive epsilon greedy')
    [success_rate_train, eps_list, _] = eval_policy(
        (alpha, gamma), env, adaptive_greedy=False)
    plt.subplot(1, 2, 1)
    plt.plot(success_rate_train)
    plt.title(
            'Q Learning Success Rate\nWithout Adaptive Epsilon Greedy(alpha = {}, gamma = {})'.format(alpha, gamma))
    plt.xlabel('Episode * 100')
    plt.ylabel('Success rate')
    plt.subplot(1, 2, 2)
    plt.plot(eps_list)
    plt.ylabel('Epsilon')
    plt.savefig('Question 22a.png')
    plt.draw()
    plt.waitforbuttonpress(timeout=5)
    plt.close()

    # with epsilon greedy
    plt.figure(figsize=(20, 10))
    print('Using adaptive epsilon greedy')
    [success_rate_train, eps_list, _] = eval_policy(
        (alpha, gamma), env, adaptive_greedy=True)
    plt.subplot(1, 2, 1)
    plt.plot(success_rate_train)
    plt.title(
            'Q Learning Success Rate\nWith Adaptive Epsilon Greedy(alpha = {}, gamma = {})'.format(alpha, gamma))
    plt.xlabel('Episode * 100')
    plt.ylabel('Success rate')
    plt.subplot(1, 2, 2)
    plt.plot(eps_list)
    plt.ylabel('Epsilon')
    plt.savefig('Question 22b.png')
    plt.draw()
    plt.waitforbuttonpress(timeout=5)
    plt.close()
