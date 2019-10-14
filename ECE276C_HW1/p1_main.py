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


def getReferencePosition(theta):
    """
    Get reference trajectory

    :param theta: float, theta
    :return: x, y position of reference trajectory
    """

    x = (0.19 + 0.02 * np.cos(4 * theta)) * np.cos(theta)
    y = (0.19 + 0.02 * np.cos(4 * theta)) * np.sin(theta)
    return x, y


def getReferenceVelocity(theta):
    """
    Get reference velocity

    :param theta: float, theta
    :return: x_dot, y_dot velocity of reference trajectory
    """
    x_dot = - np.sin(theta)*(np.cos(4*theta)/50 + 19/100) - \
        (2*np.sin(4*theta)*np.cos(theta))/25
    y_dot = np.cos(theta)*(np.cos(4*theta)/50 + 19/100) - \
        (2*np.sin(4*theta)*np.sin(theta))/25
    return x_dot, y_dot


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
    plt.savefig(title.replace('.', '').strip())
    plt.show()


if __name__ == '__main__':
    # get trajectory
    dt = 0.005
    traj = []
    for theta in np.arange(-np.pi, np.pi, dt):
        traj.append(list(getReferencePosition(theta)))
    traj = np.array(traj)

    # Problem 4
    print('\n=====Problem 4=====\n')

    # setup environment
    env = gym.make("ReacherPyBulletEnv-v0")
    env.reset()
    robot = ReacherEnv(env)
    # TODO: comment this out for random start
    robot.setJointPosition(np.pi, 0)

    # define gains
    Kp = np.diag([200, 200])
    Kd = np.diag([5, 5])

    # init control loop
    controller = ReacherEnvController(Kp, Kd)   # init controller
    traj_control = []   # placeholder for controlled trajectory
    err_list = []   # placeholder for error at each point

    # control loop
    for theta in np.arange(-np.pi, np.pi, dt):
        state_ref = getReferencePosition(theta)
        velocity_ref = getReferenceVelocity(theta)
        q0, q0_dot, q1, q1_dot = robot.getJointPositionAndVelocity()
        state_now = getForwardModel(q0, q1)[:2]
        velocity_now = (getJacobian(q0, q1) @ np.array(
            [q0_dot, q1_dot]).reshape(2, -1))[:2, 0]

        # error
        state_err = state_ref - state_now
        velocity_err = velocity_ref - velocity_now

        # PD control
        q_output = controller.pdControlEndEffector(
            state_err, velocity_err, q0, q1)

        # robot action
        env.step(q_output)

        # save trajectory
        traj_control.append(state_now.tolist())
        err_list.append(np.linalg.norm(state_err))

    # plot
    plotHelper(traj_ref=traj, traj_control=np.array(traj_control),
               title='Question 1.4 End Effector Position PD Control')

    # plot error for tunning controller
    fig = plt.figure()
    plt.plot(err_list)
    plt.title('Question 1.4 Error vs Time')
    plt.xlabel('time')
    plt.ylabel('Error (l2 norm)')
    plt.show()

    # mse
    print('Problem 4 MSE: ', calculateMSE(traj, traj_control))

    # Problem 5
    print('\n=====Problem 5=====\n')

    # reset environment
    env.reset()
    robot = ReacherEnv(env)
    robot.setJointPosition(np.pi, 0)  # TODO: comment this out for random start

    # define gains
    # Kp = np.diag([4.2, 2])
    # Kd = np.diag([0.4, 0.25])
    Kp = np.diag([4.2, 2.4])
    Kd = np.diag([0.4, 0.18])

    # init control loop
    controller = ReacherEnvController(Kp, Kd)
    traj_control = []
    err_list_q0 = []
    err_list_q1 = []
    q0, _, q1, _ = robot.getJointPositionAndVelocity()

    # control loop
    for theta in np.arange(-np.pi, np.pi, dt):
        state_ref = getReferencePosition(theta)
        velocity_ref = getReferenceVelocity(theta)
        q_ref = getIK(state_ref[0], state_ref[1], joint_angle_init_guess=np.array(
            [q0, q1]), eps=1e-4, max_iter=500)
        qdot_ref = np.dot(np.linalg.pinv(getJacobian(
            q_ref[0], q_ref[1])[:2, :2]), velocity_ref)
        q0, q0_dot, q1, q1_dot = robot.getJointPositionAndVelocity()

        # error
        q_err = q_ref - np.array([q0, q1])
        qdot_err = qdot_ref - np.array([q0_dot, q1_dot])

        # PD control
        q_output = controller.pdControlJoint(q_err, qdot_err)

        env.step(q_output)

        # save trajectory
        traj_control.append(getForwardModel(q0, q1)[:2].tolist())
        err_list_q0.append(q_err[0])
        err_list_q1.append(q_err[1])

    # plot
    traj_control = np.array(traj_control)
    plotHelper(traj_ref=traj, traj_control=traj_control,
               title='Question 1.5 Joint Angle PD Control')

    # plot error for tunning controller
    fig = plt.figure()
    plt.plot(err_list_q0)
    plt.plot(err_list_q1)
    plt.title('Question 1.5 Error vs Time')
    plt.xlabel('time')
    plt.ylabel('Error')
    plt.legend(['q0', 'q1'])
    plt.show()

    # mse
    print('Problem 5 MSE: ', calculateMSE(traj, traj_control))
