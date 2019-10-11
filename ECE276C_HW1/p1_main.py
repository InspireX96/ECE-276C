# -*- coding: utf-8 -*-
"""
ECE 276C HW1 P1 main program

Created on Mon Oct  7 11:52:05 2019

@author: xumw1
"""
import logging
import numpy as np

import gym
import pybulletgym.envs

from p1_env import *
from p1_utility import *
from p1_controller import *


def getTrajectory(theta):
    """
    Get reference trajectory

    :param theta: float, theta
    :return: x, y position of reference trajectory
    """

    x = (0.19 + 0.02 * np.cos(4 * theta)) * np.cos(theta)
    y = (0.19 + 0.02 * np.cos(4 * theta)) * np.sin(theta)
    return x, y


if __name__ == '__main__':
    # get trajectory
    dt = 0.1
    traj = []
    for theta in np.arange(-np.pi, np.pi, dt):
        traj.append(list(getTrajectory(theta)))
    traj = np.array(traj)

    # setup environment
    try:
        env = gym.make("ReacherPyBulletEnv-v0")
    except Exception:
        logging.warning('Env already registered: {}'.format(env))

    env.reset()
    robot = ReacherEnv(env)

    # define gains
    # Kp = 1
    # Kd = 10
    Kp = np.diag([1, 1])
    Kd = np.diag([10, 10])

    # Problem 4
    controller = ReacherEnvController(Kp, Kd)
    traj_control = []
    for theta in np.arange(-np.pi, np.pi, dt):
        state_ref = getTrajectory(theta)
        q0, _, q1, _ = robot.getJointPositionAndVelocity()
        state_now = getForwardModel(q0, q1)[:2]

        # error
        state_err = state_ref - state_now

        # PD control
        q_delta = controller.pdControlEndEffector(state_err, q0, q1, dt)

        # robot action
        q0, q1 = q0 + q_delta[0], q1 + q_delta[1]
        robot.setJointPosition(q0, q1)

        # save trajectory
        traj_control.append(state_now.tolist())

    traj_control = np.array(traj_control)
    # plot
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(traj_control[:, 0], traj_control[:, 1])
    plt.title('Question 1.4 End Effector PD Control')
    plt.legend(['ref traj', 'real traj'])
    plt.show()

    # Problem 5
    # TODO
