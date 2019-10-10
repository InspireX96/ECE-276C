#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gym environment wrappers for p1
"""


class ReacherEnv(object):
    """
    Environment for ReacherPyBullet, a 2-DOF arm robot
    """

    def __init__(self, env):
        """
        :param env: object, gym reacher environment
        """
        self.env = env

    def setJointPosition(self, q0, q1):
        """
        Set joint position

        :param q0: float, central joint angle (rad)
        :param q1: float, elbow joint angle (rad)
        """
        self.env.unwrapped.robot.central_joint.reset_position(q0, 0)
        self.env.unwrapped.robot.elbow_joint.reset_position(q1, 0)

    def getJointPositionAndVelocity(self):
        """
        Get joint position and angular velocity

        :returns: float. q0: central joint angle (rad)
                         q0_dot: central joint angular velocity (rad/s)
                         q1: elbow joint angle (rad)
                         q1_dot: elbow joint angular velocity (rad/s)
        """
        q0, q0_dot = self.env.unwrapped.robot.central_joint.current_position()
        q1, q1_dot = self.env.unwrapped.robot.elbow_joint.current_position()
        return q0, q0_dot, q1, q1_dot
