# -*- coding: utf-8 -*-
"""
ECE 276C HW1 P1 main program

Created on Mon Oct  7 11:52:05 2019

@author: xumw1
"""
import logging
import numpy as np

import gym
try:
    import pybulletgym.envs
except Exception as err:
    logging.warning('Should not reload pybulletgym: {}.'.format(err))

from p1_env import ReacherEnv


if __name__ == '__main__':
    try:
        env = gym.make("ReacherPyBulletEnv-v0")
    except Exception:
        logging.warning('Env already registered: {}'.format(env))
    
    state = env.reset()
    robot = ReacherEnv(env)
    print(robot.getJointPositionAndVelocity())
    robot.setJointPosition(0, 0)
    print(robot.getJointPositionAndVelocity())
    print(env.unwrapped.robot.fingertip.get_position())
    robot.setJointPosition(np.pi/2, np.pi/4)
    print(robot.getJointPositionAndVelocity())
#    robot.setJointPosition(0, 0)