""" Learn a policy using DDPG for the reach task"""
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
from torch.autograd import Variable

import gym
import pybullet
import pybulletgym.envs

import matplotlib.pyplot as plt

np.random.seed(1000)    # TODO: change random seed


# A function to soft update target networks
def weighSync(target_model, source_model, tau=0.001):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


# Write the ReplayBuffer
class Replay():
    """
    Replay buffer
    """

    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        :param: buffer_size: Size of replay buffer
        :param: init_length : Initial number of transitions to collect
        :param: state_dim : Size of the state space
        :param: action_dim : Size of the action space
        :param: env : gym environment object
        """
        self.buffer_size = buffer_size  # max size of buffer
        self.init_length = init_length  # initialize buffer using random action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        self._buffer = deque()  # replay buffer

        if self.env is not None:    # init replay buffer
            self._init_buffer()

    def _init_buffer(self):
        """
        Initialize buffer using random actions
        """
        i = 0
        break_flag = False
        while not break_flag:
            done = False
            state = self.env.reset()
            step = 1
            while not done:
                action = self.env.action_space.sample()     # sample random action
                state_next, reward, done, _ = self.env.step(action)
                step += 1

                exp = {'state': state,
                       'action': action,
                       'reward': reward,
                       'state_next': state_next,
                       'done': done}
                if step == 150:
                    exp['done'] = False      # NOTE: 150 is not done
                self.buffer_add(exp)

                state = state_next
                i += 1

                if i >= self.init_length:
                    break_flag = True
                    break

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        :param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        # buffer is not full
        self._buffer.append(exp)

        # buffer is full
        if len(self._buffer) > self.buffer_size:
            self._buffer.popleft()

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        :param: N : Number of samples to obtain from the buffer
        """
        return random.sample(self._buffer, N)


class Actor(nn.Module):
    """
    Actor Network
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the actor network

        :param: state_dim : Size of the state space
        :param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # define network layers
        hidden_size_1 = 400
        hidden_size_2 = 300

        self.fc1 = nn.Linear(self.state_dim, hidden_size_1)
        # self.ln1 = nn.LayerNorm(hidden_size_1)

        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        # self.ln2 = nn.LayerNorm(hidden_size_2)

        self.fc3 = nn.Linear(hidden_size_2, self.action_dim)

        # init weights
        self.fc1.weight.data.uniform_(-1/np.sqrt(self.state_dim),
                                      1/np.sqrt(self.state_dim))
        self.fc1.weight.data.uniform_(-1/np.sqrt(hidden_size_1),
                                      1/np.sqrt(hidden_size_1))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Define the forward pass

        param: state: The state of the environment
        """
        if isinstance(state, np.ndarray):
            x = torch.FloatTensor(state)
        else:
            x = state
        x = self.fc1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.tanh(x)


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

        self.fc1 = nn.Linear(self.state_dim, hidden_size_1)
        # self.ln1 = nn.LayerNorm(hidden_size_1)

        self.fc2 = nn.Linear(hidden_size_1 + self.action_dim, hidden_size_2)
        # self.ln2 = nn.LayerNorm(hidden_size_2)

        self.fc3 = nn.Linear(hidden_size_2, 1)

        # init weights
        self.fc1.weight.data.uniform_(-1/np.sqrt(self.state_dim),
                                      1/np.sqrt(self.state_dim))
        self.fc2.weight.data.uniform_(-1/np.sqrt(hidden_size_1 + self.action_dim),
                                      1/np.sqrt(hidden_size_1 + self.action_dim))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Define the forward pass of the critic

        :param state: The state of the environment
        :param action: action
        """
        if isinstance(state, np.ndarray):
            x = torch.FloatTensor(state)
        else:
            x = state
        x = self.fc1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        if isinstance(action, np.ndarray):
            y = torch.FloatTensor(action)
        else:
            y = action
        x = torch.cat((x, y), 1)
        x = self.fc2(x)
        # x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


# Implement a DDPG class
class DDPG():
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

        # Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        # Make sure that both networks have the same initial weights

        # Create a critic and critic_target object
        self.critic = Critic(state_dim, action_dim)
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
        action_mean = self.actor(state).detach().numpy()
        noise = np.random.multivariate_normal(
            mean=[0, 0], cov=np.diag([0.1, 0.1]))
        action = action_mean + noise
        return np.clip(action, -1, 1)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

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

        state_batch = torch.cat(state_batch).reshape(batch_size, -1)
        action_batch = torch.cat(action_batch).reshape(batch_size, -1)
        reward_batch = torch.cat(reward_batch).reshape(batch_size, -1)
        state_next_batch = torch.cat(state_next_batch).reshape(batch_size, -1)
        not_done_batch = torch.cat(not_done_batch).reshape(batch_size, -1)

        # predict next action and value
        action_next_batch = self.actor_target.forward(state_next_batch)
        target_Q = self.critic_target.forward(
            state_next_batch, action_next_batch)

        target_Q = reward_batch + \
            (self.gamma * not_done_batch * target_Q)  # TODO

        current_Q = self.critic.forward(state_batch, action_batch)

        # backward
        self.optimizer_critic.zero_grad()
        value_loss = F.mse_loss(current_Q, target_Q)
        value_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        policy_loss = - self.critic(state_batch,
                                    self.actor(state_batch)).mean()
        policy_loss.backward()
        self.optimizer_actor.step()

        # soft update
        self.update_target_networks()

        return value_loss.item(), policy_loss.item()

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations

        :param num_steps: The number of steps to train the policy for
        :returns: list, value loss, policy loss and reward over steps
        """
        time_start = time.time()
        # init
        value_loss_list = []
        policy_loss_list = []
        reward_list = []
        average_reward_list = []

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
            value_loss, policy_loss = self.update_network(batch)

            value_loss_list.append(value_loss)
            policy_loss_list.append(policy_loss)
            reward_list.append(reward)

            if i % 1000 == 0:
                print('step [{}/{}] ({:.1f} %), value_loss: {}, policy_loss: {}, average reward: {}'
                      .format(i, num_steps, i / num_steps * 100, value_loss, policy_loss, np.mean(reward_list)))

        print('Training time: {} (sec)'.format(time.time() - time_start))
        return value_loss_list, policy_loss_list, reward_list


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='DDPG train and test')
    parser.add_argument('--test', action='store_true', help='test policy')
    args = parser.parse_args()

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

    if not args.test:
        # Train the policy
        value_loss_list, policy_loss_list, reward_list = ddpg_object.train(200000)

        # plot loss
        plt.figure()
        plt.plot(value_loss_list)
        plt.plot(policy_loss_list)
        plt.xlabel('steps')
        plt.legend(['value loss', 'policy loss'])
        plt.show()

        # plot reward
        plt.figure()
        plt.plot(reward_list)
        plt.xlabel('steps')
        plt.ylabel('rewards')
        plt.show()

        # save final actor network
        with open('ddpg_actor.pkl', 'wb') as pickle_file:
            pickle.dump(ddpg_object.actor, pickle_file)
        np.save('ddpg_value_loss.npy', np.array(value_loss_list))
        np.save('ddpg_policy_loss.npy', np.array(policy_loss_list))
        np.save('ddpg_rewards.npy', np.array(reward_list))

    if args.test:
        # Evaluate the final policy
        print('\n*** Evaluating Policy ***\n')

        # load policy
        try:
            with open('ddpg_actor.pkl', 'rb') as pickle_file:
                ddpg_object.actor = pickle.load(pickle_file)

            state = env.reset()
            step = 0
            done = False
            while not done:
                action = ddpg_object.actor(state).detach().squeeze().numpy()
                next_state, r, done, _ = env.step(action)
                # env.render()
                time.sleep(0.1)
                state = next_state
                step += 1
                print('Step: {}, action: {}, reward: {}'.format(step, action, r))

        except IOError as err:
            print('ERROR: cannot load policy. Please train first. ', err)

        

        # TODO: plot average return across steps (200000), subsample using 1000 steps to compare it with last HW
        # TODO: use GPU