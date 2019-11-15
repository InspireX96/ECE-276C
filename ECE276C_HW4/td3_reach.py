""" Learn a policy using TD3 for the reach task"""
import random
import time
import copy
import pickle
import argparse
from collections import deque
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gym
import pybullet
import pybulletgym.envs

import matplotlib.pyplot as plt

from ddpg_reach import weighSync, Replay, Actor

RAND_SEED = 1000
np.random.seed(RAND_SEED)    # TODO: change random seed

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device :', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic network

        :param: state_dim : Size of the state space
        :param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # define network layers
        hidden_size_1 = 400
        hidden_size_2 = 300

        # define network layers for the twin Q-functions
        # Q1
		self.l1 = nn.Linear(state_dim + action_dim, hidden_size_1)
		self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.l3 = nn.Linear(hidden_size_2, 1)

		# Q2
		self.l4 = nn.Linear(state_dim + action_dim, hidden_size_1)
		self.l5 = nn.Linear(hidden_size_1, hidden_size_2)
		self.l6 = nn.Linear(hidden_size_2, 1)


	def forward(self, state, action):
        """
        Define the forward pass of the critic

        :param state: The state of the environment
        :param action: action
        :returns: estimated values
        """
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
        """
        Forward pass of Q1 only
        
        :param state: The state of the environment
        :param action: action
        :return: estimated value
        """
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3():
    def __init__(
            self,
            env,
            state_dim,
            action_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
    ):
        """
        :param: env: An gym environment
        :param: state_dim: Size of state space
        :param: action_dim: Size of action space
        :param: critic_lr: Learning rate of the critic
        :param: actor_lr: Learning rate of the actor
        :param: gamma: The discount factor
        :param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.test_env = copy.deepcopy(env)  # environment for evaluation only

        raise NotImplementedError


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='TD3 train and test')
    parser.add_argument('--test', action='store_true', help='test policy')
    args = parser.parse_args()

    # Define the environment
    if args.test:
        rand_init = True    # random init when testing policy
    else:
        rand_init = False
    print('\n*** Env rand init = {}    Using random seed = {} ***\n'.format(rand_init, RAND_SEED))
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1",
                   rand_init=rand_init)

    if args.test:
        env.render()    # weird render bug, needs to render here

    td3_object = TD3(
        env,
        8,
        2,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )
