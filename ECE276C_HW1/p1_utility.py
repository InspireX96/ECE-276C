#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for p1
"""

import numpy as np

# define links
L0 = 0.1
L1 = 0.11


def getDhMatrix(alpha, a, theta, d):
    """
    Calculate D-H matrix using D-hH parameters

    :param alpha: float, alpha(i-1) in D-H parameter (rad)
    :param a: float, a(i-1) in D-H parameter (m)
    :param theta: float, theta in D-H parameter (rad)
    :param d: float, d in D-H parameter (m)
    :return: 4 x 4 np array, D-H matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0, a],
                     [np.sin(theta)*np.cos(alpha), np.cos(theta) *
                      np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                     [np.sin(theta)*np.sin(alpha), np.cos(theta) *
                      np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
                     [0, 0, 0, 1]])


def getForwardModel(q0, q1):
    """
    Calculate forward kinematics using D-H parameters,
    return end effector position and orientation

    :param q0: float, central joint angle (rad)
    :param q1: float, elbow joint angle (rad)
    :return: np array, [end effector x position (m), end effector y position (m), end effector orientation (theta)] (m)
    """
    # define D-H parameters
    # columns: alpha, a, theta, d
    dh_parameters = np.array([[0, 0, q0, 0],
                              [0, L0, q1, 0],
                              [0, L1, 0, 0]])

    # calculate D-H matrices
    dh_matrix_list = []
    for i in range(dh_parameters.shape[0]):
        dh_matrix_list.append(getDhMatrix(dh_parameters[i, 0],
                                          dh_parameters[i, 1],
                                          dh_parameters[i, 2],
                                          dh_parameters[i, 3]))

    # calculate transformation matrix
    tf_matrix = dh_matrix_list[0]@dh_matrix_list[1]@dh_matrix_list[2]

    return np.array([tf_matrix[0, 3], tf_matrix[1, 3], np.arccos(tf_matrix[0, 0])])


def getJacobian(q0, q1):
    """
    Jacobian matrix is obtained by matlab helper script
    """
    return np.array([[- np.sin(q0)/10 - (11*np.cos(q0)*np.sin(q1))/100 - (11*np.cos(q1)*np.sin(q0))/100, - (11*np.cos(q0)*np.sin(q1))/100 - (11*np.cos(q1)*np.sin(q0))/100],
                     [np.cos(q0)/10 + (11*np.cos(q0)*np.cos(q1))/100 - (11*np.sin(q0)*np.sin(q1)) /
                      100, (11*np.cos(q0)*np.cos(q1))/100 - (11*np.sin(q0)*np.sin(q1))/100],
                     [1, 1]])


def getIK(x, y):
    """
    """
    pass
