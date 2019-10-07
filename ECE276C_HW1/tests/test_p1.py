# -*- coding: utf-8 -*-
"""
Unit tests for p1
"""

import sys
import logging
import numpy as np

import gym
try:
    import pybulletgym.envs
except Exception as err:
    logging.warning('Should not reload pybulletgym: {}.'.format(err))

sys.path.append('../')
from p1_env import ReacherEnv
from p1_utility import getForwardModel


def testGetForwardModel():
    print('Testing getForwardModel()')
    env = gym.make("ReacherPyBulletEnv-v0")
    env.reset()
    robot = ReacherEnv(env)
    
    for _ in range(5):
        joint_angle = np.random.rand(2)
    
        # verfy using environment observation
        robot.setJointPosition(joint_angle[0], joint_angle[1])
        ee_position_calculated = getForwardModel(joint_angle[0], joint_angle[1])[:2]
        ee_position_env = env.unwrapped.robot.fingertip.get_position()[:2]
        
        print('Joint angle: {}\nCalculated end effector position: {}\nEnv end effector position: {}\n'\
              .format(joint_angle, ee_position_calculated, ee_position_env))
        assert (ee_position_calculated.round(2) == ee_position_env.round(2)).all()

    
if __name__ == '__main__':
    testGetForwardModel()
    logging.error('Please use pytest to run test script')