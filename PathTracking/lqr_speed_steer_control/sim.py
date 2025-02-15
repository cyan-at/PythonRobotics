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

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from spliner import *

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
    goal,
    T,
    tv):
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

    debug1 = [0.0]
    debug2 = [0.0]
    debug3 = [0.0]
    debug4 = [0.0]
    debug5 = [0.0]
    debug6 = [0.0]

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

    full_T = dt * len(cx)
    if T <= 0.0:
        T = full_T
        print("updated T: %.3f" % (T))
    # else:
        # T = min(T, full_T)
    print("T!", T)

    while T >= time:
        # tv = min(0.2, time * 1) # ramp up to tv

        dl, target_ind, state.e, state.e_th, ai, expected_yaw, fb =\
            lqr_speed_steering_control(
            state, state.e, state.e_th,
            dt,
            cx, cy, cyaw,
            # ck,
            # speed_profile,
            lqr_Q, lqr_R, L, tv, time,
            distance_traveled, cumsums)

        distance_traveled += state.v * dt

        dyaw = state.v / L * math.tan(dl) * dt

        old_yaw = state.yaw

        state = update(L, state, ai, dl)

        if np.abs(dyaw) > np.pi / 4:
            print_color_str("yaw update %.3f, (%.3f, %.3f) -> %.3f" % (
                old_yaw, dl, dyaw, state.yaw), bcolors.BG_RED)
        else:
            print("yaw update %.3f, (%.3f, %.3f) -> %.3f" % (
                old_yaw, dl, dyaw, state.yaw))

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

        debug1.append(state.e_th)
        debug2.append(expected_yaw)
        debug3.append(fb)
        debug4.append(dyaw)
        debug5.append(dl)
        debug6.append(state.e)

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

        print("")

    # import ipdb; ipdb.set_trace()

    return t, x, y, yaw, v, v_msg, w_msg, ais, travel, debug1, debug2, debug3, debug4, debug5, debug6


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

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def main():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('file',
        type=str, help='file', default="/home/charlieyan1/Dev/jim/pymap2d/sdf_path_0000220.npy")
    parser.add_argument('--T',
        type=float, default=-1.0)
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

    ##########################

    # cx = ax
    # cy = ay
    # cyaw = ayaw

    ##########################

    # bx, by, byaw, bk, s = cubic_spline_planner.calc_spline_course(
    #     ax, ay, ds=0.1)
    # cx = bx
    # cy = by
    # cyaw = byaw

    ##########################

    cx = interpolate(ax, 40)
    cy = interpolate(ay, 40)

    # INTERPOATION ROTAIONS SUCKS!
    cyaw = [(x + np.pi) % (2 * np.pi) - np.pi for x in ayaw]
    cyaw = rotation_smooth(cyaw)
    # this is key
    # interpolating between -3.14 and 3.14 through 0 is spatially wrong
    cyaw = interpolate(cyaw, 40)

    ##########################

    # key_times = [x * 40* dt for x in range(len(ax))]
    # key_rots = R.from_euler('z', ayaw, degrees=False)
    # slerp = Slerp(key_times, key_rots) 
    # sample_times = np.linspace(0.0, max(key_times), len(cx))
    # interp_rots = slerp(sample_times)
    # cyaw = interp_rots.as_euler('xyz', degrees=False)[:, -1]
    # print(len(cyaw))

    ##########################

    # cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
    #     ax, ay, ds=1.0)

    ##########################

    # velocities = [[0.0, 0.0]]
    # i = 0
    # while i < len(ax) - 1:
    #     dx = ax[i+1] - ax[i]
    #     dy = ay[i+1] - ay[i]
    #     velocities.append([dx, dy])
    #     i += 1
    # # velocities.append([0.0, 0.0])

    # # import ipdb; ipdb.set_trace()

    # ctrl_pts = [CtrlPt({
    #     0: np.array([ax[i], ay[i], 0.0]),
    #     # 1: np.array([velocities[i][0], velocities[i][1], 0.0])
    #     }) for i in range(len(ax))]
    # spliner = Spliner() # create a new spliner on each call
    # s_array, s_data = spliner.process(0, 8, ctrl_pts)

    # order = s_data['order']
    # num_dof = s_data['num_dof']
    # time, state = reformat_deriv_major(
    #     order,
    #     num_dof,
    #     s_array,
    #     s_data)
    # pos_order = state[0]
    # cx = pos_order[:, 0]
    # cy = pos_order[:, 1]

    # plt.plot(ax, ay, "xb", label="waypoints")
    # plt.plot(cx, cy, "-r", label="target course")
    # # plt.plot(x, y, "-g", label="tracking")
    # plt.scatter(ax, ay)
    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()
    # plt.show()

    # import ipdb; ipdb.set_trace()

    ##########################

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

    tv = 0.2

    t, x, y, yaw, v, v_msg, w_msg, ais, distance_traveled, debug1, debug2, debug3, debug4, debug5, debug6 = do_simulation(
        state,
        cx, cy, cyaw,
        # ck, sp,
        goal,
        args.T,
        tv)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.scatter(ax, ay)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.title(args.file)

        fig, axs = plt.subplots(2, sharex=True)

        axs[0].plot(t, v, label="speed")
        axs[0].plot(t, v_msg, c='r', label="v_msg")
        axs[0].plot(t, w_msg, c='g', label="w_msg")
        axs[0].plot(t, ais, c='b', label="ais")
        # axs[0].plot(t, debug1, c='m', label="debug1")
        # axs[0].plot(t, yaw, c='k', label="yaw")

        # plt.plot(t, distance_traveled, c='b', label="distance_traveled")
        axs[0].grid(True)
        # axs[0].xlabel("Time [sec]")
        # axs[0].ylabel("Speed [m/s]")
        axs[0].legend()

        axs[1].plot(
            # [dt * i for i in range(len(cyaw))],
            # # [np.rad2deg(iyaw) for iyaw in cyaw],
            # cyaw,
            t,
            debug2,
            "-r",
            label="expected yaw")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            yaw,
            c='k',
            label="yaw")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug1,
            c='b',
            label="th_e")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug3,
            c='g',
            label="debug3")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug4,
            c='c',
            label="debug4")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug5,
            c='y',
            label="debug5")
        axs[1].plot(
            # list(range(len(cyaw))),
            # yaw[:len(cyaw)],
            t,
            debug6,
            '*c',
            label="debug6")

        # plt.plot(list(range(len(cyaw))), debug1, "-r", label="debug1")
        axs[1].grid(True)
        axs[1].legend()
        # axs[1].xlabel("line length[m]")
        # axs[1].ylabel("yaw angle[deg]")

        plt.show()

if __name__ == '__main__':
    main()
