#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECE 276C HW1 P2 main program
"""

import numpy as np
from racecar.SDRaceCar import SDRaceCar

def dynamic_model(v, gamma, theta, d):
    """
    Dynamic model
    
    :param v: float, velocity
    :param gamma: float, front wheel angle
    :param theta: float, vehicle angle
    :param d: float, front and rear wheel distance
    """
    v_x = v * np.cos(gamma) * np.cos(theta)
    v_y = v * np.cos(gamma) * np.sin(theta)
    d_theta = v / d * np.sin(gamma)

    return np.array([v_x, v_y, d_theta])

def kinematic_model(pos_ref, pos_now, theta_now):
    """[summary]
    
    :param pos_ref: [description]
    :param pos_now: [description]
    """
    x_d, y_d = pos_ref[0], pos_ref[1]
    x, y = pos_now[0], pos_now[1]

    gamma = np.arctan2((y_d - y), (x_d - x)) - theta_now
    return gamma

if __name__ == '__main__':
    env = SDRaceCar(render_env=True, track="FigureEight")
    env.reset()
    env.render()

    # action space: [wheel angle, thrust]
    # observation space: [x, y, theta, vx, vy, d_theta, h]
    # where h is the coordinate on the track the car has to reach

    wheel_distance = env.l_f + env.l_r

    # define
    Kp = 1e-1

    for i in range(500):
        # entering control loop
        state_obs = env.get_observation()
        state_ref = state_obs[-1]     # desired position
        
        x_now, y_now, theta_now = state_obs[0], state_obs[1], state_obs[2]
        
        dx, dy = state_ref[0] - x_now, state_ref[1] - y_now

        gamma = kinematic_model(pos_ref=state_ref, pos_now=np.array([x_now, y_now]), theta_now=theta_now)

        d_body = np.dot(np.array([[np.cos(theta_now), -np.sin(theta_now)],
                                  [np.sin(theta_now), np.cos(theta_now)]]),
                        np.array([dx, dy]))
        print('d_body: ', d_body)

        print('gamma:', gamma)
        if abs(gamma) >= 0.2:
            thrust = -1 + np.random.rand() * 0.1
        else:
            thrust = Kp * (d_body[1] ** 2 - d_body[0] ** 2)     # TODO
        # regularize
        thrust -= 0.5
        thrust = max(thrust, -0.9)
        thrust = min(thrust, 0.5)
        # if i < 10:  # TODO
        #     thrust = 1
        # elif i < 50:
        #     thrust = -0.5
        # else:
        #     thrust = - 1
        env.step([gamma, thrust])
        print('i: {}, input (gamma, thrust): {}'.format(i, [gamma, thrust]))
        env.render()