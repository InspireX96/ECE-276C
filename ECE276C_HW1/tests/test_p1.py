#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for p1
Please use pytest to run this script automatically
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
from p1_utility import getForwardModel, getJacobian, getIK

def testGetForwardModel():
    print('\n=====Testing getForwardModel()=====\n')
    env = gym.make("ReacherPyBulletEnv-v0")
    env.reset()
    robot = ReacherEnv(env)

    for _ in range(5):
        joint_angle = np.random.rand(2)

        # verify using environment observation
        robot.setJointPosition(joint_angle[0], joint_angle[1])
        # don't know how to test theta, env return nonsense value
        ee_position_calculated = getForwardModel(
            joint_angle[0], joint_angle[1])[:2]
        ee_position_env = env.unwrapped.robot.fingertip.get_position()[:2]
        # ee_orientation_env = env.unwrapped.robot.fingertip.get_orientation()

        print('Joint angle: {}\nCalculated end effector position: {}\nEnv end effector position: {}'
              .format(joint_angle, ee_position_calculated, ee_position_env))
        assert (ee_position_calculated.round(3)
                == ee_position_env.round(3)).all()


def testGetJacobian():
    print('\n=====Testing getJacobian()=====\n')
    # Because for some reason, gym always returns 0 tip velocity even we input joint velocity,
    # here we define a simple test case to test Jacobian
    joint_angle = np.array([0, 0])
    jacobian_matrix = getJacobian(joint_angle[0], joint_angle[1])
    joint_velocity = np.array([0.5, 0])

    x_dot_calculated = jacobian_matrix@joint_velocity
    print('Joint angle: {}\nJoint velocity: {}\n'
          'Calculated end effector velocity: {}'
          .format(joint_angle, joint_velocity, x_dot_calculated))
    assert (x_dot_calculated.round(3) == np.array(
        [0, 0.21*0.5, 0.5]).round(3)).all()


def testGetIK():
    print('\n=====Testing getIK()=====\n')
    env = gym.make("ReacherPyBulletEnv-v0")
    env.reset()
    robot = ReacherEnv(env)

    ee_position_list = np.array([[0.05, 0.02],
                                 [0.1, 0.05],
                                 [0.2, 0.0],
                                 [-0.02, 0.08],
                                 [-0.1, 0.04]])
    for ee_position in ee_position_list:
        q_calculated = getIK(ee_position[0], ee_position[1])
        robot.setJointPosition(q_calculated[0], q_calculated[1])
        ee_position_env = env.unwrapped.robot.fingertip.get_position()[:2]
        print('Input end effector position: {}\n'
              'End effector position using joint angles calculated by IK: {}'
              .format(ee_position, ee_position_env))
        assert (ee_position.round(2) == ee_position_env.round(2)).all()


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.StreamHandler()
        ])
    logging.getLogger()

    # run test manually
    testGetForwardModel()
    testGetJacobian()
    testGetIK()
    print('All test passed')
