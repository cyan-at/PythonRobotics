#! /usr/bin/env python3

"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

./sim.py /home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000210.npy
./sim.py /home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000280.npy

"""
from lqr import *

import matplotlib.pyplot as plt

from PathPlanning.CubicSpline import cubic_spline_planner

import argparse

show_animation = True

def update(L, state, a, delta):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state

def do_simulation(
    state,
    cx, cy, cyaw,
    # ck, speed_profile,
    goal):
    T = 100.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05

    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    v_msg = [0.0]
    w_msg = [0.0]
    ais = [0.0]
    t = [0.0]
    travel = [0.0]

    distances = [] # len(cx) - 1
    cumsums = [0.0] # len(cx)
    i = 0
    pt = np.array([cx[i], cy[i]])
    while i < len(cx) - 1:
        new_pt = np.array([cx[i+1], cy[i+1]])
        dist = np.linalg.norm(new_pt - pt, ord=2)

        distances.append(dist)
        cumsums.append(cumsums[-1] + dist)

        i += 1
        pt = new_pt
    cumsums = np.array(cumsums)

    time = 0.0
    distance_traveled = 0.0
    while T >= time:
        tv = 0.2 # go slow
        # tv = min(0.2, time * 1) # ramp up to tv

        dl, target_ind, state.e, state.e_th, ai =\
            lqr_speed_steering_control(
            state, state.e, state.e_th,
            dt,
            cx, cy, cyaw,
            # ck,
            # speed_profile,
            lqr_Q, lqr_R, L, tv, time,
            distance_traveled, cumsums)

        distance_traveled += state.v * dt

        state = update(L, state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        v_msg_i = v_msg[-1] + ai * dt
        w_msg_i = dl * dt

        v_msg.append(v_msg_i)
        w_msg.append(w_msg_i)
        ais.append(ai)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        travel.append(distance_traveled)

        print("state.v", state.v)

        # if target_ind % 1 == 0 and show_animation:
        #     plt.cla()
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect(
        #         'key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        #     plt.plot(cx, cy, "-r", label="course")
        #     plt.plot(x, y, "ob", label="trajectory")
        #     plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
        #               + ",target index:" + str(target_ind))
        #     plt.pause(0.0001)

    # import ipdb; ipdb.set_trace()

    return t, x, y, yaw, v, v_msg, w_msg, ais, travel


def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)

    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    # for i in range(40):
    #     speed_profile[-i] = target_speed / (50 - i)
    #     if speed_profile[-i] <= 1.0 / 3.6:
    #         speed_profile[-i] = 1.0 / 3.6

    return speed_profile

def two_d_rvec_vec_from_matrix_2d(m):
    x = m[0, 2]
    y = m[1, 2]

    atan2 = np.arctan2(m[1, 0], m[0, 0])
    # SUPER #COOL #IMPORTANT #port
    # this is how we get the original theta 2019-08-02
    # https://stackoverflow.com/a/32549077
    # deal with quadrant logic

    '''
    theta_0 = math.acos(m[0, 0])
    theta_1 = math.asin(m[1, 0])
    theta_2 = math.asin(-1.0 * m[0, 1])
    theta_3 = math.acos(m[1, 1])
    # TODO(someone) finish this up,
    # do not always all match, sometimes negative
    thetas = [theta_0, theta_1, theta_2, theta_3]
    mode_thetas = Util.modes(thetas)
    if len(mode_thetas) > 0:
        # print("more than one theta found", mode_thetas)
        pass
    '''

    return [x, y, atan2], None

def main():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('file',
        type=str, help='file', default="/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000220.npy")
    args = parser.parse_args()

    payload = np.load(
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000001.npy',
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000055.npy',
        # '/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000240.npy',
        args.file,
        allow_pickle=True).tolist()
    all_homs = payload['path']
    xythetas = np.array(
        [two_d_rvec_vec_from_matrix_2d(x)[0] for x in all_homs])
    ax = xythetas[:, 0]
    ay = xythetas[:, 1]
    ayaw = xythetas[:, 2]

    cx = ax
    cy = ay
    cyaw = ayaw

    # bx, by, byaw, bk, s = cubic_spline_planner.calc_spline_course(
    #     ax, ay, ds=0.1)
    # cx = bx
    # cy = by
    # cyaw = byaw

    # import ipdb; ipdb.set_trace()

    cx = interpolate(ax, 10)
    cy = interpolate(ay, 10)
    cyaw = interpolate(ayaw, 10)

    print("LQR steering control tracking start!!")
    # ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    goal = [cx[-1], cy[-1]]


    # target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s
    # sp = calc_speed_profile(cyaw, target_speed)

    state = State(
        x=cx[0],
        y=cy[0],
        yaw=cyaw[0],
        v=0.0)

    t, x, y, yaw, v, v_msg, w_msg, ais, distance_traveled = do_simulation(
        state,
        cx, cy, cyaw,
        # ck, sp,
        goal)

    # import ipdb; ipdb.set_trace()

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        plt.plot(t, v, label="speed")
        plt.plot(t, v_msg, c='r', label="v_msg")
        plt.plot(t, w_msg, c='g', label="w_msg")
        plt.plot(t, ais, c='b', label="ais")
        # plt.plot(t, distance_traveled, c='b', label="distance_traveled")
        plt.grid(True)
        plt.xlabel("Time [sec]")
        plt.ylabel("Speed [m/s]")
        plt.legend()

        # plt.subplots(1)
        # plt.plot(list(range(len(cyaw))), [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        # plt.grid(True)
        # plt.legend()
        # plt.xlabel("line length[m]")
        # plt.ylabel("yaw angle[deg]")

        plt.show()


if __name__ == '__main__':
    main()
