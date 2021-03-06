"""
Unit test for replay buffer
Please use pytest to run this script automatically
"""

import sys
import numpy as np

import gym
import pybullet
import pybulletgym.envs

sys.path.append('../')

from ddpg_reach_skeleton import Replay


def test_buffer_add():
    """
    Test buffer add function
    """
    print('\n===== Testing buffer add =====\n')
    # init params
    buffer_size = 10

    replay_buffer = Replay(buffer_size=buffer_size,
                           init_length=10,
                           state_dim=8,
                           action_dim=2,
                           env=None)

    # test normal add
    state = 0
    for i in range(buffer_size):
        action = np.random.rand()
        state_next, reward, done = i, i, False

        # test buffer add using toy example
        exp = {'state': state,
               'action': action,
               'reward': reward,
               'state_next': state_next,
               'done': done}
        replay_buffer.buffer_add(exp)

        state = state_next

        assert len(replay_buffer._buffer) == i + 1
        assert replay_buffer._buffer[-1] == exp

    print('items in buffer: ', replay_buffer._buffer)

    # test overflow
    state = 999
    for i in range(3):
        action = np.random.rand()
        state_next, reward, done = i * buffer_size, i * buffer_size, True

        # test buffer add
        exp = {'state': state,
               'action': action,
               'reward': reward,
               'state_next': state_next,
               'done': done}
        replay_buffer.buffer_add(exp)

        state = state_next

        assert len(replay_buffer._buffer) == buffer_size
        assert replay_buffer._buffer[-1] == exp

    print('items in buffer: ', replay_buffer._buffer)


def test_buffer_sample():
    """
    Test buffer sample function
    """
    print('\n===== Testing buffer sample =====\n')
    # init params
    buffer_size = 10

    replay_buffer = Replay(buffer_size=buffer_size,
                           init_length=10,
                           state_dim=8,
                           action_dim=2,
                           env=None)

    # add stuff into buffer
    state = 0
    for i in range(buffer_size):
        action = np.random.rand()
        state_next, reward, done = i, i, False

        # test buffer add using toy example
        exp = {'state': state,
               'action': action,
               'reward': reward,
               'state_next': state_next,
               'done': done}
        replay_buffer.buffer_add(exp)

        state = state_next

    print('items in buffer: ', replay_buffer._buffer)

    for i in range(1, buffer_size // 2):
        result = replay_buffer.buffer_sample(i)
        print('sample result: {}, type: {}'.format(result, type(result)))

        assert len(result) == i


def test_buffer_init():
    """
    Test buffer initialization
    """
    print('\n===== Testing buffer initialization =====\n')
    # init params
    buffer_size = 5000
    init_length = 500

    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    replay_buffer = Replay(buffer_size=buffer_size,
                           init_length=init_length,
                           state_dim=8,
                           action_dim=2,
                           env=env)
    print('buffer length: ', len(replay_buffer._buffer))
    assert len(replay_buffer._buffer) == init_length


if __name__ == '__main__':
    print('Testing replay buffer')
    test_buffer_add()
    test_buffer_sample()
    test_buffer_init()
    print('All tests passed')
