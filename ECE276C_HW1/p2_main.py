#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECE 276C HW1 P2 main program
"""

import numpy as np
from matplotlib import pyplot as plt
from racecar.SDRaceCar import SDRaceCar


def dynamic_model(v, gamma, theta, d):
    """
    Dynamic model

    :param v: float, velocity
    :param gamma: float, front wheel angle
    :param theta: float, vehicle angle
    :param d: float, front and rear wheel distance
    :return: np array (len=3), [x_velocity, y_velocity, angular_velocity]
    """
    v_x = v * np.cos(gamma) * np.cos(theta)
    v_y = v * np.cos(gamma) * np.sin(theta)
    d_theta = v / d * np.sin(gamma)
    return np.array([v_x, v_y, d_theta])


def kinematic_model(pos_ref, pos_now, theta_now):
    """
    Kinematic model to get wheel angle

    :param pos_ref: np array (len=2), reference position
    :param pos_now: np array (len=2), current position
    :return: float, wheel angle (rad)
    """
    x_d, y_d = pos_ref[0], pos_ref[1]
    x, y = pos_now[0], pos_now[1]

    gamma = np.arctan2((y_d - y), (x_d - x)) - theta_now
    if gamma <= - np.pi:
        gamma += 2 * np.pi
    elif gamma >= np.pi:
        gamma -= 2 * np.pi
    return gamma


def plotHelper(traj_ref, traj_control, title):
    """
    Plots for Question 2

    :param traj_ref: N x 2 np ndarray, reference trajectory
    :param traj_control: N x 2 np ndarray, real trajectory
    :param title: plot title
    """
    assert isinstance(traj_ref, np.ndarray) and isinstance(
        traj_control, np.ndarray)
    assert isinstance(title, str)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.plot(traj_ref[:, 0], traj_ref[:, 1])
    plt.plot(traj_control[:, 0], traj_control[:, 1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['ref traj', 'real traj'])
    plt.ioff()
    plt.savefig(title.strip())
    plt.show()


if __name__ == '__main__':
    # TODO: select track here
    # track = "FigureEight"
    # track = "Linear"
    track = "Circle"

    # setup environment
    env = SDRaceCar(render_env=True, track=track)
    env.reset()

    # action space: [wheel angle, thrust]
    # observation space: [x, y, theta, vx, vy, d_theta, h]
    # where h is the coordinate on the track the car has to reach

    wheel_distance = env.l_f + env.l_r

    # define
    Kp = 1e-2
    launch_control_step = 10    # steps to perform launch control, 10 for low error, 16 for fastest lap
    speed_penalty_limit = 6     # max speed limit before penalty, 6 for low error, 8.1 for fastest lap

    # init
    steps = 0
    previous_ind = 0
    done = False
    traj_reference = []     # placeholder for reference trajectory
    traj_control = []   # placeholder for controlled trajectory

    # control loop
    while not done:
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
        gamma = kinematic_model(pos_ref=state_ref, pos_now=np.array(
            [x_now, y_now]), theta_now=theta_now)

        # thrust control
        if steps < launch_control_step:
            print(' launch control')
            thrust = 1  # launch control
        elif vel > speed_penalty_limit:
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
        _, _, done = env.step([gamma, thrust])
        steps += 1
        print('steps: {}, vel: {}, input (gamma, thrust): {}'.format(
            steps, vel, [gamma, thrust]))
        # env.render()  # TODO: comment this out to disable rendering

        # check lap completion
        current_ind = env.closest_track_ind
        if current_ind - previous_ind <= -500:
            done = True
        previous_ind = current_ind

        # save trajectory
        traj_reference.append(state_ref)
        traj_control.append(np.array([x_now, y_now]))

    print('Number of steps: ', steps)

    # plot
    plotHelper(traj_ref=np.array(traj_reference), traj_control=np.array(traj_control),
               title='Question 2 Result Track {}'.format(track))
