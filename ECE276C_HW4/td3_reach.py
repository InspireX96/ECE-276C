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

        :param state_dim : Size of the state space
        :param action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # define network layers
        hidden_size_1 = 400
        hidden_size_2 = 300

        # define network layers for the twin Q-functions
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)

        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.fc5 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc6 = nn.Linear(hidden_size_2, 1)

        # init weights
        self.fc1.weight.data.uniform_(-1/np.sqrt(self.state_dim + self.action_dim),
                                      1/np.sqrt(self.state_dim + self.action_dim))
        self.fc2.weight.data.uniform_(-1/np.sqrt(hidden_size_1),
                                      1/np.sqrt(hidden_size_1))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        # init weights
        self.fc4.weight.data.uniform_(-1/np.sqrt(self.state_dim + self.action_dim),
                                      1/np.sqrt(self.state_dim + self.action_dim))
        self.fc5.weight.data.uniform_(-1/np.sqrt(hidden_size_1),
                                      1/np.sqrt(hidden_size_1))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Define the forward pass of the critic

        :param state: The state of the environment
        :param action: action
        :returns: estimated values
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        """
        Forward pass of Q1 only

        :param state: The state of the environment
        :param action: action
        :return: estimated value
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
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
            policy_freq=2
    ):
        """
        :param env: An gym environment
        :param state_dim: Size of state space
        :param action_dim: Size of action space
        :param critic_lr: Learning rate of the critic
        :param actor_lr: Learning rate of the actor
        :param gamma: The discount factor
        :param batch_size: The batch size for training
        :param policy_freq: policy update frequency
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.test_env = copy.deepcopy(env)  # environment for evaluation only
        self.policy_freq = policy_freq

        # Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # Make sure that both networks have the same initial weights

        # Create a critic and critic_target object
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # Make sure that both networks have the same initial weights

        # Define the optimizer for the actor
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Define the optimizer for the critic
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        # define a replay buffer
        self.ReplayBuffer = Replay(buffer_size=10000,
                                   init_length=1000,
                                   state_dim=state_dim,
                                   action_dim=action_dim,
                                   env=env)

    def select_action_with_noise(self, state):
        """
        Select action with exploration noise
        Returned action will be clipped to [-1, 1]

        :param state: The state of the environment
        :return: np array, action with Gaussian noise
        """
        action_mean = self.actor(torch.FloatTensor(
            state).to(device)).cpu().detach().numpy()
        noise = np.random.multivariate_normal(
            mean=[0, 0], cov=np.diag([0.1, 0.1]))
        action = action_mean + noise
        return np.clip(action, -1, 1)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor, tau=5e-3)
        weighSync(self.critic_target, self.critic, tau=5e-3)

    def update_network(self, batch):
        """
        A function to update the function just once

        :param batch: list, minibatch samples
        :returns: float, value and policy loss
        """
        # parse batch
        batch_size = len(batch)
        state_batch = []
        action_batch = []
        reward_batch = []
        state_next_batch = []
        not_done_batch = []
        for exp in batch:
            state_batch.append(torch.FloatTensor(exp['state']))
            action_batch.append(torch.FloatTensor(exp['action']))
            reward_batch.append(torch.FloatTensor([exp['reward']]))
            state_next_batch.append(torch.FloatTensor(exp['state_next']))
            not_done_batch.append(torch.FloatTensor([not exp['done']]))

        state_batch = torch.cat(state_batch).reshape(batch_size, -1).to(device)
        action_batch = torch.cat(action_batch).reshape(
            batch_size, -1).to(device)
        reward_batch = torch.cat(reward_batch).reshape(
            batch_size, -1).to(device)
        state_next_batch = torch.cat(state_next_batch).reshape(
            batch_size, -1).to(device)
        not_done_batch = torch.cat(not_done_batch).reshape(
            batch_size, -1).to(device)

        # get next action
        action_next_batch = torch.clamp(self.actor_target(state_next_batch) + torch.FloatTensor(np.random.multivariate_normal(
            mean=[0, 0], cov=np.diag([0.1, 0.1]), size=len(state_next_batch))), -1, 1)  # next action batch with noise
        
        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(state_next_batch, action_next_batch)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward_batch + not_done_batch * self.gamma * target_Q
        # TODO
        import ipdb; ipdb.set_trace()

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations

        :param num_steps: The number of steps to train the policy for
        :returns: list, critic loss, actor loss over steps;
                  step and average reward over each evaluation during training
        """
        time_start = time.time()
        # init
        critic_loss_list = []
        actor_loss_list = []

        # placeholders for policy eval during training
        eval_step_list = []
        eval_average_reward_list = []

        traj_step = 1
        state = self.env.reset()

        for i in range(num_steps):
            # NOTE: add noise to action using multivariable Gaussian and clip it between -1 to 1
            action = self.select_action_with_noise(state)
            state_next, reward, done, _ = self.env.step(action)     # step
            traj_step += 1

            # store transition in R
            exp = {'state': state,
                   'action': action,
                   'reward': reward,
                   'state_next': state_next,
                   'done': done}
            if traj_step == 150:
                exp['done'] = False      # NOTE: 150 is not done
            self.ReplayBuffer.buffer_add(exp)

            # move on
            state = state_next

            if done:
                traj_step = 1
                state = self.env.reset()

            # sample minibatch
            batch = self.ReplayBuffer.buffer_sample(self.batch_size)

            # update network
            critic_loss, actor_loss = self.update_network(batch)


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

    td3_object = TD3(env, 8, 2)

    if not args.test:
        # Train the policy
        critic_loss_list, actor_loss_list, eval_step_list, eval_average_reward_list = td3_object.train(
            200)
