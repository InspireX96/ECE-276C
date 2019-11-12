""" Learn a policy using DDPG for the reach task"""
import random

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

np.random.seed(1000)


# TODO: A function to soft update target networks
def weighSync(target_model, source_model, tau=0.001):
    raise NotImplementedError


# TODO: Write the ReplayBuffer
class Replay():
    """
    Replay buffer
    """
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: buffer_size: Size of replay buffer
        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length  # TODO: what is this
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        self._buffer = deque()  # replay buffer

    # TODO: Complete the function
    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        # buffer is not full
        self._buffer.append(exp)

        # buffer is full
        if len(self._buffer) > self.buffer_size:
            self._buffer.popleft()

    # TODO: Complete the function
    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        return random.sample(self._buffer, N)


class Actor(nn.Module):
    """
     Actor Network
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # define network layers
        self.fc1 = nn.Linear(self.state_dim, 128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, action)

        # init weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = state
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.tanh(x)


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # define network layers
        self.fc_state = nn.Linear(self.state_dim, 64)
        self.ln_state = nn.LayerNorm(64)

        self.fc_action = nn.Linear(self.action_dim, 64)
        self.ln_action = nn.LayerNorm(64)

        self.fc_out = nn.Linear(64, 1)

        # init weights
        self.fc_state.weight.data.normal_(0, 0.1)
        self.fc_action.weight.data.normal_(0, 0.1)
        self.fc_out.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        x = state
        x = self.fc_state(x)
        x = self.ln_state(x)

        y = action
        y = self.fc_action(y)
        y = self.ln_action(y)

        out = F.relu(x + y)
        return self.fc_out(out)


# TODO: Implement a DDPG class
class DDPG():
    def __init__(
            self,
            env,
            action_dim,
            state_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env

        # Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        # Make sure that both networks have the same initial weights

        # Create a critic and critic_target object
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # Make sure that both networks have the same initial weights

        # Define the optimizer for the actor
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Define the optimizer for the critic
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # TODO: define a replay buffer
        self.ReplayBuffer = None

    # TODO: Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    # TODO: Complete the function
    def update_network(self):
        """
        A function to update the function just once
        """
        raise NotImplementedError

    # TODO: Complete the function
    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    ddpg_object = DDPG(
        env,
        8,
        2,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )
    # Train the policy
    ddpg_object.train(100)

    # Evaluate the final policy
    state = env.reset()
    done = False
    while not done:
        action = ddpg_object.actor(state).detach().squeeze().numpy()
        next_state, r, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        state = next_state
