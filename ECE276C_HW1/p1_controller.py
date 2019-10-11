#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD controllers for p1
"""
import numpy as np
from matplotlib import pyplot as plt
import gym
try:
    import pybulletgym.envs
except Exception as err:
    logging.warning('Should not reload pybulletgym: {}.'.format(err))

from p1_env import ReacherEnv
from p1_utility import getForwardModel, getJacobian, getIK


def getTrajectory(theta):
    x = (0.19 + 0.02 * np.cos(4 * theta)) * np.cos(theta)
    y = (0.19 + 0.02 * np.cos(4 * theta)) * np.sin(theta)
    return x, y


if __name__ == '__main__':
    # get traj
    traj = []
    for theta in np.arange(-np.pi, np.pi, 0.1):
        traj.append(list(getTrajectory(theta)))
    traj = np.array(traj)
    # plot
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.plot(traj[:, 0], traj[:, 1])
    # plt.show()

    try:
        env = gym.make("ReacherPyBulletEnv-v0")
    except Exception:
        logging.warning('Env already registered: {}'.format(env))
    
    # for theta in np.arange(-np.pi, np.pi, 0.1):
    env.reset()
    robot = ReacherEnv(env)

    # define
    Kp = 1

    traj_control = []
    for theta in np.arange(-np.pi, np.pi, 0.1):
        state_ref = getTrajectory(theta)
        q0, _, q1, _ = robot.getJointPositionAndVelocity()
        state_now = getForwardModel(q0, q1)[:2]

        state_err = state_ref - state_now
        state_delta = Kp * state_err    # P control

        j_mat = getJacobian(q0, q1)[:2, :2]
        q_delta = (np.linalg.pinv(j_mat) @ state_err.reshape(2, -1))[:, 0]

        # robot action
        q0, q1 = q0 + q_delta[0], q1 + q_delta[1]
        robot.setJointPosition(q0, q1)

        traj_control.append(state_now.tolist())
    
    traj_control = np.array(traj_control)
    # plot
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(traj_control[:, 0], traj_control[:, 1])
    plt.show()
