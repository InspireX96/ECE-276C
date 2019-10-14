#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD controllers for p1
"""
import logging
import numpy as np

from p1_utility import getForwardModel, getJacobian, getIK


class ReacherEnvController():
    """
    Controllers for reacher environment
    """

    def __init__(self, Kp=1, Kd=10):
        """
        :param Kp: float or 2 x 2 np ndarray, P gain, default is 1
        :param Ki: float or 2 x 2 np ndarray, D gain, default is 10
        """
        self.Kp = Kp
        self.Kd = Kd
        logging.info('Initialized controller with Kp: {}, Kd: {}'.format(Kp, Kd))

    def pdControlEndEffector(self, state_err, velocity_err, q0, q1):
        """
        PD controller using error in end-effector space

        :param state_err: np array (len=2), end effector error
        :param velocity_err: np array (len=2), end effector velocity error
        :parma q0: float, central joint angle (rad)
        :param q1: float, elbow joint angle (rad)
        :return q_output: np array (len=2), joint angle control output
        """
        assert isinstance(state_err, np.ndarray) and isinstance(velocity_err, np.ndarray)
        assert state_err.shape == (2, ) and velocity_err.shape == (2, )

        # P control
        state_delta = np.dot(self.Kp, state_err)

        # D control
        d_control_input = velocity_err
        d_control_output = np.dot(self.Kd, d_control_input)

        # combine PD
        state_delta += d_control_output

        # kinematics
        j_mat = getJacobian(q0, q1)[:2, :2]
        q_output = (j_mat.T @ state_delta.reshape(2, -1))[:, 0]  # force control

        return q_output

    def pdControlJoint(self, q_err, qdot_err):
        """
        PD controller using error in joint space

        :param q_err: np array (len=2), joint angle error
        :param qdot_error: np array (len=2), joint angular velocity error
        :return q_output: np array (len=2), joint angle control output
        """
        assert isinstance(q_err, np.ndarray) and isinstance(qdot_err, np.ndarray)
        assert q_err.shape == (2, ) and qdot_err.shape == (2, )

        # P control
        q_delta = np.dot(self.Kp, q_err)

        # D control
        d_control_output = np.dot(self.Kd, qdot_err)

        # combine PD
        q_output = q_delta + d_control_output

        return q_output
