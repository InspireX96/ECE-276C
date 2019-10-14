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
    if gamma <= - np.pi:
        gamma += 2 * np.pi
    elif gamma >= np.pi:
        gamma -= 2 * np.pi
    # gamma = np.arctan((y_d - y) / (x_d - x)) - theta_now
    return gamma

if __name__ == '__main__':
    env = SDRaceCar(render_env=True, track="FigureEight")
    # env = SDRaceCar(render_env=True, track="Linear")
    # env = SDRaceCar(render_env=True, track="Circle")
    env.reset()
    env.render()

    # action space: [wheel angle, thrust]
    # observation space: [x, y, theta, vx, vy, d_theta, h]
    # where h is the coordinate on the track the car has to reach

    wheel_distance = env.l_f + env.l_r

    # define
    Kp = 1e-2

    # control loop
    for i in range(500):
        # observation
        state_obs = env.get_observation()
        state_ref = state_obs[-1]     # desired position
        [x_now, y_now, theta_now, v_x, v_y] = state_obs[:5]
        
        dx, dy = state_ref[0] - x_now, state_ref[1] - y_now
        d_body = np.dot(np.array([[np.cos(theta_now), -np.sin(theta_now)],
                                  [np.sin(theta_now), np.cos(theta_now)]]),
                        np.array([dx, dy]))
        vel = np.sqrt(v_x ** 2 + v_y ** 2)

        # gamma control
        gamma = kinematic_model(pos_ref=state_ref, pos_now=np.array([x_now, y_now]), theta_now=theta_now)

        # thrust control
        if i < 15:
            print(' launch control')
            thrust = 1  # launch control
        elif vel > 6:
            print(' speed penalty')
            thrust = -1     # speed penalty
        else:
            if abs(gamma) >= 0.15:
                print('  wheel angle penalty')
                thrust = -1 + np.random.rand() * 0.05   # wheel angle penalty
            else:
                thrust = Kp * (d_body[1] ** 2 - d_body[0] ** 2)     # TODO
                # speed limit
                thrust -= 0.5
                print('  thrust before speed limit: ', thrust)
                thrust = max(thrust, -1)      # lower limit
                thrust = min(thrust, -0.5)      # upper limit

        # action
        env.step([gamma, thrust])
        print('i: {}, vel: {}, input (gamma, thrust): {}'.format(i, vel, [gamma, thrust]))
        env.render()