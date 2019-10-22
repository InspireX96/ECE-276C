"""
ECE 276C HW2 Question 2
"""
import gym
import numpy as np
from p1_main import testPolicy

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
        testPolicy(env, policy, trials=100, verbose=False)

    # Q 2.1(b)
    print('\n===== Question 2.1(b) =====\n')
    alpha = 0.05
    for gamma in [0.9, 0.95, 0.99]:
        print('alpha: {}, gamma: {}'.format(alpha, gamma))
        qlearn = QlearningPolicy(env, alpha=alpha, gamma=gamma)
        policy = qlearn.trainPolicy()
        testPolicy(env, policy, trials=100, verbose=False)

    # Q 2.2
    # TODO