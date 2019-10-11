# -*- coding: utf-8 -*-
"""
ECE 276C HW1 P1 main program

Created on Mon Oct  7 11:52:05 2019

@author: xumw1
"""
import logging
import numpy as np
from matplotlib import pyplot as plt

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

def calculateMSE(traj_ref, traj_control):
    """
    Calculate mean square error

    :param traj_ref: np ndarray, reference trajectory
    :param traj_control: np ndarray, real trajectory
    :return: mean square error
    """
    traj_delta = traj_ref - traj_control
    temp = np.sum(traj_delta ** 2, axis=1)
    mse = temp.sum() / traj_ref.shape[0]
    return mse

def plotHelper(traj_ref, traj_control, title):
    """
    Plots for Question 1

    :param traj_ref: np ndarray, reference trajectory
    :param traj_control: np ndarray, real trajectory
    :param title: plot title
    """
    assert isinstance(title, str)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.plot(traj_ref[:, 0], traj_ref[:, 1])
    plt.plot(traj_control[:, 0], traj_control[:, 1])
    plt.title(title)
    plt.legend(['ref traj', 'real traj'])
    plt.show()
    # TODO: save fig


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

    # plot
    traj_control = np.array(traj_control)
    plotHelper(traj_ref=traj, traj_control=traj_control, title='Question 1.4 End Effector Position PD Control')
    
    # mse
    print('Problem 4 MSE: ', calculateMSE(traj, traj_control))

    # Problem 5
    # reset environment
    env.reset()
    robot = ReacherEnv(env)

    # define gains
    Kp = np.diag([1, 1])
    Kd = np.diag([1e-5, 1e-5])

    # Problem 4
    controller = ReacherEnvController(Kp, Kd)
    traj_control = []
    for theta in np.arange(-np.pi, np.pi, dt):
        state_ref = getTrajectory(theta)
        q_ref = getIK(state_ref[0], state_ref[1])
        q0, _, q1, _ = robot.getJointPositionAndVelocity()

        # error
        q_err = q_ref - np.array([q0, q1])

        # PD control
        q_delta = controller.pdControlJoint(q_err, dt)

        # robot action
        q0, q1 = q0 + q_delta[0], q1 + q_delta[1]

        # unwrap
        q0, q1 = q0 % (2 * np.pi), q1 % (2 * np.pi)

        # robot action
        robot.setJointPosition(q0, q1)

        # save trajectory
        traj_control.append(getForwardModel(q0, q1)[:2].tolist())

    # plot
    traj_control = np.array(traj_control)
    plotHelper(traj_ref=traj, traj_control=traj_control, title='Question 1.5 Joint Angle PD Control')

    # mse
    print('Problem 5 MSE: ', calculateMSE(traj, traj_control))