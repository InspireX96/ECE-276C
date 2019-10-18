"""
ECE 276C HW2 Question 2
"""
import logging
import gym
import numpy as np


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
        self.state_space = np.arange(env.observation_space.n)
        self.action_space = np.arange(env.action_space.n)

        # init Q function
        self.Q_table = np.random.rand(
            env.observation_space.n, env.action_space.n)
        logging.warning('Q table shape: {}'.format(self.Q_table.shape))

    def updateQ(self, state, state_next, action, reward):
        """
        Q learning update

        :param state: int, current state
        :param state_next: int, next state
        :param action: int, action
        :param reward: float, reward
        """
        Q_current = self.Q_table[state][action]
        Q_next = self.Q_table[state]    # this is array

        # update Q function
        self.Q_table[state][action] += self.alpha * \
            (reward + self.gamma * np.max(Q_next) - Q_current)

    def _takeAction(self, state, episode):
        """
        Take action with epsilon-greedy
        
        :param state: [description]
        :param episode: [description]
        """
        pass

    def trainPolicy(self, episode=5000):
        """
        Train policy

        :param episode: int (>0), training episode
        """
        assert isinstance(episode, int) and episode > 0
        
        for i in range(episode):
            state = self.env.reset()
            done = False
            pass

    def testPolicy(self):
        """
        Test policy
        """
        pass


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()
    policy = QlearningPolicy(env, alpha=0.05, gamma=0.99)
