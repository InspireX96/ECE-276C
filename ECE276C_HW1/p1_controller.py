#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD controllers for p1
"""
import logging
import numpy as np
from matplotlib import pyplot as plt

from p1_env import ReacherEnv
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
        # define register for D control diff
        self._state_error_memory = np.zeros(2)

    def pdControlEndEffector(self, state_err, q0, q1, dt):
        """
        PD controller using teh error in the end-effector

        :param state_err: np array (len=2), state error
        :parma q0: float, central joint angle (rad)
        :param q1: float, elbow joint angle (rad)
        :param dt: float, time step
        :return q_delta: np array (len=2), joint angle difference
        """
        # P control
        if isinstance(self.Kp, np.ndarray):
            state_delta = self.Kp @ state_err
        else:
            state_delta = self.Kp * state_err

        # D control
        d_control_input = (state_err - self._state_error_memory) / dt
        if isinstance(self.Kd, np.ndarray):
            d_control_output = self.Kd @ d_control_input
        else:
            d_control_output = self.Kd * d_control_input
        self.state_err_memory = state_err   # update register

        # combine PD
        state_delta += d_control_output

        # kinematics
        j_mat = getJacobian(q0, q1)[:2, :2]
        q_delta = (np.linalg.pinv(j_mat) @ state_err.reshape(2, -1))[:, 0]
        return q_delta
